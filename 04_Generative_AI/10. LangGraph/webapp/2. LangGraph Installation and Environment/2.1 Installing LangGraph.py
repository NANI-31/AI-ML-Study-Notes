import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="2.1 Installing LangGraph - LangGraph Curriculum",
    page_icon="📦",
    layout="wide"
)

# Custom Theme CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-cmd {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE PACKAGE GENERATOR CLASS FOR WORKBENCH & PIPELINE
# ==============================================================================
class DependencyManifestGenerator:
    """
    Generates deterministic, conflict-free package installation scripts
    for enterprise LangGraph deployment environments.
    """
    def __init__(self):
        self.package_map = {
            "Core Graph Runtime": ["langgraph", "langgraph-checkpoint-postgres"],
            "LangChain Primitives": ["langchain-core", "langchain"],
            "Observability SDK": ["langsmith"],
            "OpenAI Partner": ["langchain-openai"],
            "Anthropic Partner": ["langchain-anthropic"],
            "Google Gemini Partner": ["langchain-google-genai"],
            "Community Tools": ["langchain-community", "tavily-python"]
        }

    def generate_commands(self, tool: str, selected_components: List[str]) -> Tuple[str, str, int]:
        all_pkgs = []
        for comp in selected_components:
            all_pkgs.extend(self.package_map.get(comp, []))

        pkg_str = " ".join(all_pkgs)
        if tool == "uv (Recommended - Rust Powered)":
            cmd = f"uv pip install {pkg_str}"
            lock_cmd = "uv lock"
            est_time_sec = max(2, len(all_pkgs) * 1)
        elif tool == "pip":
            cmd = f"pip install {pkg_str}"
            lock_cmd = "pip freeze > requirements.txt"
            est_time_sec = max(10, len(all_pkgs) * 5)
        else:
            cmd = f"conda install -c conda-forge {pkg_str}"
            lock_cmd = "conda env export > environment.yml"
            est_time_sec = max(25, len(all_pkgs) * 8)

        return cmd, lock_cmd, est_time_sec


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("2.1 Installing LangGraph: Package Managers, Virtual Envs, & Modular SDKs")
st.caption("Chapter 2: LangGraph Installation and Environment | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
Setting up a professional production environment for **LangGraph** requires more than running a blind `pip install`. 
Modern AI engineering demands deterministic dependency resolution, pinned partner packages, isolated virtual environments, 
and high-speed Rust-powered tooling.

In this lesson, you will master:
* Installing LangGraph with standard `pip` versus blazing-fast Rust-powered `uv`.
* Creating isolated virtual runtimes to prevent system Python pollution.
* Decoupling `langgraph`, `langchain-core`, and partner provider libraries (`langchain-openai`, `langchain-anthropic`).
* Installing LangSmith telemetry clients and prebuilt enterprise community toolkits.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Modern LangGraph Tooling & Dependency Pipeline", divider="rainbow")

st.markdown("""
The architecture below outlines the enterprise setup flow: from virtual environment initialization, 
to Rust-powered resolution with `uv`, to modular partner packages, through to CI/CD verification.
""")

blueprint_dot = """
digraph ToolingPipeline {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_env {
        label = "1. Runtime Isolation Tier";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        PythonBase [label="Python 3.10 - 3.12 Runtime\\n(Standard Enterprise Host)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        VirtualEnv [label="Isolated Virtual Environment (.venv)\\n(Created via uv venv or python -m venv)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_resolver {
        label = "2. High-Speed Dependency Resolver";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        UvTool [label="uv Package Manager (Rust)\\n(10x - 100x Faster Resolution | Global Cache)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        LockFile [label="Deterministic Lockfile (uv.lock)\\n(Cryptographic SHA-256 Hash Verification)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_layers {
        label = "3. Decoupled Modular Packages";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        CoreGraph [label="langgraph\\n(StateGraph + Pregel Runtime)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        LangSmithPkg [label="langsmith\\n(Telemetry & Tracing Client)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        PartnerPkgs [label="Partner Packages\\n(langchain-openai, langchain-anthropic)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        ToolsPkgs [label="Toolkits & Adapters\\n(langchain-community, tavily-python)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_ci {
        label = "4. Containerization & Production Build";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        DockerImage [label="Production Docker Container\\n(Multi-stage build, minimal attack surface)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    PythonBase -> VirtualEnv [label="1. Init Runtime"];
    VirtualEnv -> UvTool [label="2. Invoke Resolver"];
    UvTool -> LockFile [label="3. Generate Hash Lock"];

    UvTool -> CoreGraph;
    UvTool -> LangSmithPkg;
    UvTool -> PartnerPkgs;
    UvTool -> ToolsPkgs;

    CoreGraph -> DockerImage [label="4. Multi-Stage Deploy"];
    PartnerPkgs -> DockerImage;
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - INSTALLING WITH PIP
# ==============================================================================
st.header("2. Installing with Pip: Pinned Releases, Wheels, & Dependency Resolution", divider="rainbow")

st.markdown("""
The universal standard for installing Python software is `pip`. Installing LangGraph via pip is straightforward:
""")

st.code("""
# Install the core LangGraph state machine runtime
pip install -U langgraph

# Install PostgreSQL checkpointer support for durable production persistence
pip install -U langgraph-checkpoint-postgres
""", language="bash")

col_pip1, col_pip2 = st.columns([1, 1])

with col_pip1:
    st.markdown("""
    ### Production Best Practice: Version Pinning
    In enterprise CI/CD pipelines, never use unconstrained `pip install langgraph`. Always pin semantic versions:
    * Pinning prevents unexpected breaking changes across minor releases:
      `langgraph>=0.2.20,<0.3.0`
    """)

with col_pip2:
    st.markdown("""
    ### Wheel Caching
    * Modern LangGraph releases distribute pre-compiled binary `.whl` (wheels), ensuring rapid installation without local C-compiler dependencies.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - INSTALLING WITH UV
# ==============================================================================
st.header("3. Installing with uv: Blazing-Fast Rust-Powered Package Management", divider="rainbow")

st.markdown("""
In 2024–2026, **`uv`** (built by Astral in Rust) replaced legacy pip as the enterprise standard for Python package management. 
`uv` is **10 to 100 times faster** than pip because it uses a global content-addressable cache and parallel network requests.
""")

col_uv1, col_uv2 = st.columns([1, 1])

with col_uv1:
    st.markdown("""
    ### Why UV Dominates AI Development
    * **Sub-Second Installation**: Installing LangGraph and partner SDKs takes under 1.5 seconds instead of 45 seconds with pip.
    * **Deterministic Resolver**: Universal resolution algorithm prevents backtracking hell when balancing complex dependencies (e.g. NumPy, PyTorch, Pydantic).
    * **Built-in Python Management**: `uv python install 3.12` downloads and manages Python runtimes directly without pyenv.
    """)

with col_uv2:
    st.markdown("""
    ### Typical UV Workflow
    """)
    st.code("""
# 1. Install uv globally
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Initialize project and virtual environment
uv init enterprise-agent
cd enterprise-agent
uv venv --python 3.12

# 3. Add LangGraph with instantaneous resolution
uv add langgraph langsmith langchain-openai
    """, language="bash")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CREATING A VIRTUAL ENVIRONMENT
# ==============================================================================
st.header("4. Creating a Virtual Environment: Isolated Runtimes with venv, uv, and Conda", divider="rainbow")

st.markdown("""
Installing packages directly into your global system Python is an antipattern that breaks operating system utilities 
and triggers library collisions between conflicting projects.
""")

env_comp = [
    {
        "Tool": "uv venv",
        "Creation Speed": "< 50 milliseconds",
        "Isolation Guarantee": "Strict isolated symlink sandbox",
        "Recommended Use": "Modern AI engineering & production Docker builds"
    },
    {
        "Tool": "python -m venv",
        "Creation Speed": "~1 - 2 seconds",
        "Isolation Guarantee": "Standard library isolation",
        "Recommended Use": "Environments where external CLI tools cannot be installed"
    },
    {
        "Tool": "Conda / Mamba",
        "Creation Speed": "~15 - 30 seconds",
        "Isolation Guarantee": "Full binary environment with CUDA/C++ runtimes",
        "Recommended Use": "Local GPU model training and fine-tuning"
    }
]
st.dataframe(pd.DataFrame(env_comp), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - INSTALLING LANGCHAIN
# ==============================================================================
st.header("5. Installing LangChain: Decoupled Core vs Integration Packages", divider="rainbow")

st.markdown("""
Do NOT install the legacy monolithic `pip install langchain` unless you specifically require standard chains. 
Instead, install only the decoupled packages needed by your graph nodes:
""")

st.code("""
# Core abstractions, prompt templates, and message schemas (lightweight)
pip install -U langchain-core

# Standard retrieval and agent chain utilities
pip install -U langchain
""", language="bash")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
By installing only `langgraph` and `langchain-core`, container image sizes drop from ~1.2 GB to **under 180 MB**, 
dramatically slashing Kubernetes pod cold-start times during auto-scaling events!
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - INSTALLING LANGSMITH
# ==============================================================================
st.header("6. Installing LangSmith: Native Tracing SDKs & Client Libraries", divider="rainbow")

st.markdown("""
LangSmith instrumentation requires installing the lightweight SDK client:
""")

st.code("""
# Install LangSmith client
pip install -U langsmith
""", language="bash")

col_ls1, col_ls2 = st.columns([1, 1])

with col_ls1:
    st.markdown("""
    ### What LangSmith Installs
    * High-performance asynchronous HTTP logging client with non-blocking thread workers.
    * OpenTelemetry (OTel) exporters and context propagators.
    * Dataset management utilities for offline evaluation.
    """)

with col_ls2:
    st.markdown("""
    ### Zero Overhead
    * If `LANGSMITH_TRACING=false` or unset, the SDK is a zero-latency no-op that introduces zero network hops or latency.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - INSTALLING MODEL PROVIDERS
# ==============================================================================
st.header("7. Installing Model Providers: Pinned Partner Packages (OpenAI, Anthropic, Bedrock)", divider="rainbow")

st.markdown("""
Install only the specific model partner packages required by your application:
""")

prov_table = [
    {"Target Provider": "OpenAI (GPT-4o, o1, Embeddings)", "Package Name": "langchain-openai", "Command": "pip install -U langchain-openai"},
    {"Target Provider": "Anthropic (Claude 3.5 Sonnet)", "Package Name": "langchain-anthropic", "Command": "pip install -U langchain-anthropic"},
    {"Target Provider": "Google (Gemini 1.5 / 2.0)", "Package Name": "langchain-google-genai", "Command": "pip install -U langchain-google-genai"},
    {"Target Provider": "AWS Bedrock (Claude on AWS)", "Package Name": "langchain-aws", "Command": "pip install -U langchain-aws boto3"},
    {"Target Provider": "Local Ollama (Llama 3.3)", "Package Name": "langchain-ollama", "Command": "pip install -U langchain-ollama"}
]
st.dataframe(pd.DataFrame(prov_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - INSTALLING TOOL INTEGRATIONS
# ==============================================================================
st.header("8. Installing Tool Integrations: Community Toolkits, MCP Packages, & Prebuilt Nodes", divider="rainbow")

st.markdown("""
Agent nodes frequently need prebuilt tools for web search, database querying, or sandbox execution:
""")

st.code("""
# Community tools & integrations
pip install -U langchain-community

# Popular production search toolkits
pip install -U tavily-python duckduckgo-search

# PostgreSQL database toolkit
pip install -U psycopg2-binary sqlalchemy
""", language="bash")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Automated Environment Bootstrap Script", divider="rainbow")

st.markdown("""
Below is an automated, idempotent bash bootstrap script for setting up an enterprise-grade LangGraph development environment:
""")

script_code = """
#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Initializing Enterprise LangGraph Runtime"
echo "=========================================="

# 1. Verify Python Version (3.10+)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# 2. Check for uv; install if missing
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv via Astral installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# 3. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment via uv..."
    uv venv .venv --python 3.12
fi

# 4. Activate Virtual Environment
source .venv/bin/activate

# 5. Install Pinned Production Packages
echo "Installing LangGraph & Partner Packages..."
uv pip install \\
    "langgraph>=0.2.20" \\
    "langgraph-checkpoint-postgres>=2.0.0" \\
    "langchain-core>=0.3.0" \\
    "langchain-openai>=0.2.0" \\
    "langsmith>=0.1.100" \\
    "tavily-python>=0.5.0"

# 6. Generate Deterministic Lockfile
uv lock

echo "=========================================="
echo "Setup Complete! Activate with: source .venv/bin/activate"
echo "=========================================="
"""
st.code(script_code, language="bash")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Package Dependency Matrix & Command Generator", divider="rainbow")

st.markdown("""
Use the tool below to generate customized, conflict-free installation commands and estimate installation 
latencies across package managers.
""")

gen = DependencyManifestGenerator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Select Packages & Tooling")
    
    sel_tool = st.selectbox(
        "Package Manager",
        ["uv (Recommended - Rust Powered)", "pip", "conda"],
        index=0
    )
    
    st.markdown("#### Select Stack Components")
    sel_comps = []
    for comp in gen.package_map.keys():
        if st.checkbox(comp, value=("Core" in comp or "OpenAI" in comp or "Observability" in comp)):
            sel_comps.append(comp)

with col_wb2:
    st.subheader("2. Generated Installation Manifest")
    
    if sel_comps:
        cmd, lock_cmd, est_sec = gen.generate_commands(sel_tool, sel_comps)
        
        m1, m2 = st.columns(2)
        m1.metric("Selected Components", len(sel_comps))
        m2.metric("Estimated Install Time", f"~{est_sec} seconds")

        st.markdown("#### Execution Command")
        st.code(cmd, language="bash")

        st.markdown("#### Lockfile Generation")
        st.code(lock_cmd, language="bash")
    else:
        st.warning("Select at least one component on the left.")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Questions on environment management and dependency pinning test an engineer's operational maturity.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: Why is `uv` preferred over standard `pip` for LangGraph enterprise deployments?"):
    st.markdown("""
    **Answer**: `uv` is written in Rust and utilizes a global content-addressable cache and parallel resolution. 
    It resolves complex LangChain dependency graphs in sub-2-second latencies compared to 30-60 seconds for pip. 
    Furthermore, `uv` produces deterministic cross-platform lockfiles (`uv.lock`) with cryptographic hashes, 
    preventing supply-chain tampering and ensuring that development, staging, and Kubernetes production pods run 
    identical byte-for-byte dependencies.
    """)

with st.expander("Q2: What is the risk of using unpinned dependencies (`pip install langgraph`) in CI/CD?"):
    st.markdown("""
    **Answer**: Unpinned dependencies allow package managers to pull the latest minor or patch version on each build. 
    If a dependency (e.g., Pydantic, LangChain-core) releases a breaking change or deprecation, production deployment builds 
    can fail unexpectedly without any application code changes. Enterprise standards require strict version pinning 
    (e.g., `langgraph==0.2.28` or lockfiles).
    """)

with st.expander("Q3: Why should production Docker images avoid installing `langchain-community` if not needed?"):
    st.markdown("""
    **Answer**: `langchain-community` is an enormous meta-package with hundreds of optional integrations (SQL drivers, 
    PDF parsers, vector connectors). Installing it unnecessarily bloats Docker image size by 1 GB+, dramatically increases 
    the vulnerability attack surface (CVE scan alerts), and slows container registry pull speeds during horizontal pod auto-scaling.
    """)

with st.expander("Q4: How do you verify that LangSmith tracing is active after installing the package?"):
    st.markdown("""
    **Answer**: 
    1. Ensure `langsmith` is installed in the virtual environment.
    2. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.
    3. Run a minimal graph invocation or script: `from langsmith import Client; Client().ping()`.
    4. Verify that a new run trace appears within 5 seconds on the LangSmith web dashboard under the configured project name.
    """)

with st.expander("Q5: What Python version range is supported and recommended for LangGraph?"):
    st.markdown("""
    **Answer**: LangGraph officially supports Python 3.10, 3.11, and 3.12. Python 3.11 or 3.12 is strongly recommended 
    due to significant CPython core performance improvements (10-25% faster bytecode execution) and modern type hinting 
    features (`TypedDict`, `Annotated`) used heavily in StateGraph schemas.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Installing LangGraph in System Python Without a Virtualenv (CRITICAL)**
    * *Error*: Running `sudo pip install langgraph` on production host instances.
    * *Impact*: Pollutes operating system package namespaces and breaks Linux system utilities (apt, yum).
    * *Fix*: Always create and activate an isolated `.venv` using `uv venv` or Docker containers.
    """)

    st.error("""
    **Trap 2: Mismatched Pydantic v1 vs v2 Dependencies (CRITICAL)**
    * *Error*: Mixing legacy packages compiled against Pydantic v1 with LangGraph (which requires Pydantic v2).
    * *Impact*: Triggers fatal import-time schema validation crashes and metaclass conflicts.
    * *Fix*: Enforce Pydantic >= 2.7.0 across all dependencies.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Omitting PostgreSQL Binary Drivers for Persistent Checkpointing (WARNING)**
    * *Warning*: Installing `langgraph-checkpoint-postgres` without `psycopg2-binary` or `asyncpg`.
    * *Impact*: Checkpointer throws `ModuleNotFoundError: No module named 'psycopg'` at connection time.
    * *Fix*: Install `psycopg[binary,pool]` alongside the checkpointer package.
    """)

    st.warning("""
    **Trap 4: Committing `.venv` Directories to Git Repositories (WARNING)**
    * *Warning*: Forgetting `.venv/` in `.gitignore`.
    * *Impact*: Pushes gigabytes of host-specific binaries to GitHub, breaking CI/CD clone speeds.
    * *Fix*: Include `.venv`, `__pycache__`, and `*.pyc` in root `.gitignore`.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: uv</strong><br><br>
        <em>Front:</em> Why is uv 10x-100x faster than pip?<br><br>
        <em>Back:</em> Written in Rust; uses global caching, concurrent downloads, and optimized dependency SAT solvers.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: Modular Partner</strong><br><br>
        <em>Front:</em> What is the OpenAI partner package for LangChain?<br><br>
        <em>Back:</em> <code>langchain-openai</code> (decoupled directly maintained integration).
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Lockfile</strong><br><br>
        <em>Front:</em> What is the role of <code>uv.lock</code>?<br><br>
        <em>Back:</em> Guarantees exact byte-for-byte reproducibility across all environments.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Postgres Saver</strong><br><br>
        <em>Front:</em> Which package provides PostgreSQL checkpointers?<br><br>
        <em>Back:</em> <code>langgraph-checkpoint-postgres</code>.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LangGraph installation:")

chk1 = st.checkbox("I know how to install LangGraph using both `pip` and high-speed `uv`.", value=True)
chk2 = st.checkbox("I understand why virtual environment isolation (`.venv`) is mandatory in enterprise software.", value=True)
chk3 = st.checkbox("I know how to install decoupled partner packages (`langchain-openai`, `langchain-anthropic`).", value=True)
chk4 = st.checkbox("I understand the role of `langsmith` for non-intrusive runtime observability.", value=True)
chk5 = st.checkbox("I know how to install database checkpointer packages for PostgreSQL persistence.", value=True)
chk6 = st.checkbox("I can generate deterministic lockfiles (`uv.lock`) for production Docker builds.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 2.1: Installing LangGraph! Proceed to Lesson 2.2: Environment Configuration.")
