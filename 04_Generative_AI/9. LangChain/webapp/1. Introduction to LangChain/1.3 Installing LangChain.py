import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.3 Installing LangChain - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.3 Installing LangChain: Environments, Packages & Secrets")
st.caption("Chapter 1: Introduction to LangChain | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Setting up a production-grade Generative AI development environment requires more than running a simple `pip install`. 
In modern enterprise stacks, developers must coordinate Python version constraints, isolated virtual environments, 
decoupled partner packages, secure API key injection, and automated installation verification.

With the release of **LangChain v0.3+**, the ecosystem abandoned monolithic package distribution. Rather than installing a 
massive 500MB library containing hundreds of unused vector databases and model clients, LangChain is partitioned into 
**`langchain-core`** alongside standalone partner packages (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`).

In this lesson, you will master:
- **Modular Distribution Topology**: Why monolithic `langchain` was split into lightweight partner packages.
- **Dependency Isolation**: Best practices with modern Python virtual environments and package managers (`uv`, `pip`).
- **Provider Integrations**: Installing and configuring OpenAI, Anthropic, and Google Gemini connectors.
- **Enterprise Secret Hygiene**: Managing environment variables and `.env` files without committing secrets to Git.
- **Automated Verification**: Running programmatic smoke tests to validate client connectivity before deployment.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class EnvironmentHealthChecker:
    """Evaluates environment configuration and generates production commands in module scope."""
    def __init__(self):
        self.provider_matrix = {
            "OpenAI": {
                "package": "langchain-openai",
                "env_var": "OPENAI_API_KEY",
                "default_model": "gpt-4o",
                "install_cmd": "pip install -U langchain-core langchain-openai"
            },
            "Anthropic": {
                "package": "langchain-anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "default_model": "claude-3-5-sonnet-20241022",
                "install_cmd": "pip install -U langchain-core langchain-anthropic"
            },
            "Google GenAI": {
                "package": "langchain-google-genai",
                "env_var": "GOOGLE_API_KEY",
                "default_model": "gemini-1.5-pro",
                "install_cmd": "pip install -U langchain-core langchain-google-genai"
            },
            "Ollama (Local)": {
                "package": "langchain-ollama",
                "env_var": "OLLAMA_HOST",
                "default_model": "llama3.1",
                "install_cmd": "pip install -U langchain-core langchain-ollama"
            }
        }

    def evaluate_configuration(self, selected_providers: List[str], use_langsmith: bool) -> Dict[str, Any]:
        packages_to_install = ["langchain-core"]
        env_vars_needed = []
        install_commands = []
        
        for prov in selected_providers:
            data = self.provider_matrix.get(prov)
            if data:
                packages_to_install.append(data["package"])
                env_vars_needed.append(data["env_var"])
                install_commands.append(data["install_cmd"])

        if use_langsmith:
            packages_to_install.append("langsmith")
            env_vars_needed.extend(["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"])

        return {
            "total_packages": len(set(packages_to_install)),
            "package_list": sorted(list(set(packages_to_install))),
            "required_env_vars": env_vars_needed,
            "recommended_install_cmd": f"pip install -U {' '.join(sorted(list(set(packages_to_install))))}",
            "status": "CONFIG_READY"
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern LangChain v0.3+ Package Distribution Topology", divider="rainbow")

dot_blueprint = """
digraph PackageTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_python_env {
        label="1. Runtime & Environment Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        python_runtime [label="Python 3.10+ Virtual Environment\\n(.venv / uv / conda)", fillcolor="#0F172A"];
        env_file [label="Environment Secret Configuration\\n(.env -> os.environ / Secrets Manager)", fillcolor="#1E293B"];
    }

    subgraph cluster_core_dist {
        label="2. Minimal Core Distribution Tier";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        core_pkg [label="langchain-core (v0.3+)\\nBaseChatModel, Runnable Protocol, Prompts, OutputParsers\\n(Zero heavy transitive dependencies)", fillcolor="#1E293B"];
        community_pkg [label="langchain-community (Optional)\\nThird-party integrations & experimental connectors", fillcolor="#1E293B"];
    }

    subgraph cluster_partner_pkgs {
        label="3. Official Partner Packages (Isolated Dependencies)";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        openai_pkg [label="langchain-openai\\n(openai SDK v1.0+)", fillcolor="#064E3B"];
        anthropic_pkg [label="langchain-anthropic\\n(anthropic SDK v0.30+)", fillcolor="#064E3B"];
        google_pkg [label="langchain-google-genai\\n(google-genai SDK)", fillcolor="#064E3B"];
    }

    subgraph cluster_app {
        label="4. Application Consumption Layer";
        fontcolor="#A855F7";
        style=dashed;
        color="#334155";
        app_code [label="Enterprise Application Logic\\nfrom langchain.chat_models import init_chat_model", fillcolor="#4C1D95"];
    }

    python_runtime -> core_pkg [label="Installs"];
    python_runtime -> partner_pkgs [label="Installs Selected"];
    env_file -> app_code [label="Injects Keys"];
    core_pkg -> app_code [label="Interfaces"];
    openai_pkg -> app_code [label="Provider Model"];
    anthropic_pkg -> app_code [label="Provider Model"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTION 2: PYTHON REQUIREMENTS ---
st.header("2. Python Requirements: Version Constraints, Virtual Environments, and Modern Package Managers", divider="rainbow")

st.markdown("""
Modern LangChain (v0.3+) officially requires **Python 3.9 through 3.12** (Python 3.10+ recommended). 
Python 3.8 reached official end-of-life and is unsupported by modern Pydantic v2 core dependencies.

### Production Environment Best Practices
Never install LangChain into your global system Python environment. Always provision an isolated environment:

```bash
# Standard venv creation
python -m venv .venv

# Activate on Linux / macOS
source .venv/bin/activate

# Activate on Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
```

For 10x faster installation and deterministic dependency resolution, enterprise engineering teams increasingly adopt 
**`uv`** (an ultra-fast Python package installer written in Rust):
```bash
uv venv
uv pip install -U langchain-core langchain-openai
```
""")

# --- SECTION 3: INSTALLING LANGCHAIN WITH PIP ---
st.header("3. Installing LangChain with Pip: The Decoupled Core Architecture", divider="rainbow")

st.markdown("""
In legacy versions, developers ran `pip install langchain`. In modern production stacks, you install only what you consume:

- **`pip install -U langchain-core`**: The foundational package containing the `Runnable` protocol, message primitives, 
  and base abstractions. It has almost zero third-party dependencies and installs in less than two seconds.
- **`pip install -U langchain`**: The main orchestration library containing common document loaders, text splitters, 
  and pre-built chains.
- **`pip install -U langchain-community`**: Contains community-contributed integrations (custom vector stores, specialized retrievers). 
  Only install if you require specific niche connectors.
""")

# --- SECTION 4: INSTALLING PROVIDER INTEGRATIONS ---
st.header("4. Installing Provider Integrations: Why Monolithic Dependencies Were Deprecated", divider="rainbow")

st.markdown("""
In 2023, LangChain faced dependency hell: upgrading the `openai` library would inadvertently break an incompatible 
version requirement in `anthropic` or `huggingface_hub`. 

To solve this, LangChain split every major provider into an isolated **Partner Package**:
- Dedicated version pinning.
- Independent release cadences.
- Minimal Docker container sizes (deploying an OpenAI service doesn't bundle AWS or HuggingFace SDKs).
""")

# --- SECTION 5: INSTALLING OPENAI INTEGRATION ---
st.header("5. Installing OpenAI Integration: langchain-openai and Model Optimization", divider="rainbow")

st.markdown("""
To use models like `gpt-4o`, `gpt-4o-mini`, and OpenAI text embeddings:

```bash
pip install -U langchain-openai
```

### Authentication Setup
Set the `OPENAI_API_KEY` environment variable in your terminal or `.env` file:
```bash
export OPENAI_API_KEY="sk-proj-..."
```
""")

# --- SECTION 6: INSTALLING ANTHROPIC INTEGRATION ---
st.header("6. Installing Anthropic Integration: langchain-anthropic and Tool Calling Support", divider="rainbow")

st.markdown("""
To use Anthropic's Claude 3.5 Sonnet and Haiku models:

```bash
pip install -U langchain-anthropic
```

### Authentication Setup
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
""")

# --- SECTION 7: INSTALLING GOOGLE INTEGRATION ---
st.header("7. Installing Google Integration: langchain-google-genai and Multimodal Pipelines", divider="rainbow")

st.markdown("""
To connect to Google Gemini 1.5 Pro and Flash:

```bash
pip install -U langchain-google-genai
```

### Authentication Setup
```bash
export GOOGLE_API_KEY="AIzaSy..."
```
""")

# --- SECTION 8: ENVIRONMENT VARIABLES ---
st.header("8. Environment Variables: Configuration Boundaries and Process Environments", divider="rainbow")

st.markdown("""
LangChain automatically inspects standard operating system environment variables when initializing provider models. 
If `OPENAI_API_KEY` is present in `os.environ`, `ChatOpenAI()` automatically reads it without requiring explicit parameter passing:

```python
# No hardcoded key required; LangChain reads from environment automatically
model = ChatOpenAI(model="gpt-4o")
```

This adheres to **Twelve-Factor App** principles, ensuring that code and configuration remain strictly separated.
""")

# --- SECTION 9: MANAGING API KEYS ---
st.header("9. Managing API Keys: Least Privilege, Secret Rotation, and Production Injection", divider="rainbow")

st.markdown("""
Enterprise secret hygiene requires three non-negotiable rules:

1. **Never Hardcode Secrets**: Hardcoding `api_key="sk-..."` in Python scripts leads to catastrophic credential leaks on GitHub.
2. **Principle of Least Privilege**: Create project-scoped API keys with restricted budget limits and restricted model permissions.
3. **Production Secret Injection**: In Kubernetes or Docker, inject keys using secret stores (AWS Secrets Manager, HashiCorp Vault) 
   rather than burning them into Docker image layers.
""")

# --- SECTION 10: WORKING WITH .ENV FILES ---
st.header("10. Working with .env Files: python-dotenv and Local Development Security", divider="rainbow")

st.markdown("""
For local development, store secrets in a `.env` file at the root of your project:

```bash
# .env (NEVER COMMIT THIS FILE TO GIT)
OPENAI_API_KEY=sk-proj-abc123xyz
ANTHROPIC_API_KEY=sk-ant-abc123xyz
GOOGLE_API_KEY=AIzaSy123xyz

# Enable LangSmith Distributed Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_abc123xyz
LANGCHAIN_PROJECT=production-rag-service
```

### Always Add .env to .gitignore
```gitignore
# .gitignore
.env
.venv/
__pycache__/
*.pyc
```

### Loading in Python
```python
from dotenv import load_dotenv

# Call at the very top of your entrypoint script
load_dotenv()
```
""")

# --- SECTION 11: VERIFYING INSTALLATION ---
st.header("11. Verifying Installation: Programmatic Smoke Tests and init_chat_model Validation", divider="rainbow")

st.markdown("""
Once installed, execute a minimal smoke test using the modern **`init_chat_model`** factory function to verify that:
1. Package dependencies are resolveable.
2. API keys are authenticated.
3. Network connectivity and TLS handshakes succeed.
""")

st.code("""
from langchain.chat_models import init_chat_model

try:
    # Factory function automatically loads appropriate partner package
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    response = model.invoke("Reply with the word 'OK' if connected.")
    print("Verification Successful:", response.content)
except Exception as e:
    print("Verification Failed:", str(e))
""", language="python")

# --- SECTION 12: PRODUCTION IMPLEMENTATION ---
st.header("12. Production Implementation: Automated Environment Health Check and Dynamic Model Initializer", divider="rainbow")

st.markdown("""
The following complete, runnable Python module acts as a self-contained diagnostic suite that audits active 
API keys, confirms partner package installations, and verifies client connectivity:
""")

st.code("""
import os
from typing import Dict, Any, List

def audit_genai_environment() -> Dict[str, Any]:
    \"\"\"Audits local environment variables for required GenAI provider credentials.\"\"\"
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "LangSmith Tracing": "LANGCHAIN_API_KEY"
    }
    
    audit_results = {}
    for provider, env_var in providers.items():
        is_configured = env_var in os.environ and len(os.environ[env_var].strip()) > 8
        audit_results[provider] = {
            "environment_variable": env_var,
            "status": "CONFIGURED" if is_configured else "MISSING",
            "masked_preview": (os.environ[env_var][:6] + "..." + os.environ[env_var][-4:]) if is_configured else "N/A"
        }
    return audit_results

# Example invocation:
# report = audit_genai_environment()
# print(report)
""", language="python")

# --- SECTION 13: INTERACTIVE WORKBENCH ---
st.header("13. Interactive Workbench: Package Dependency and Environment Configurator", divider="rainbow")

st.markdown("""
Select the providers and observability tools required for your application to generate the optimal, minimal 
`pip` installation command and `.env` template.
""")

checker = EnvironmentHealthChecker()

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.subheader("Select Model Providers")
    prov_openai = st.checkbox("OpenAI (gpt-4o, gpt-4o-mini)", value=True)
    prov_anthropic = st.checkbox("Anthropic (claude-3-5-sonnet)", value=True)
    prov_google = st.checkbox("Google (gemini-1.5-pro)", value=False)
    prov_ollama = st.checkbox("Ollama (Local Llama 3.1)", value=False)

with col_cfg2:
    st.subheader("Observability & Telemetry")
    prov_langsmith = st.checkbox("Enable LangSmith Distributed Tracing", value=True)

selected_list = []
if prov_openai: selected_list.append("OpenAI")
if prov_anthropic: selected_list.append("Anthropic")
if prov_google: selected_list.append("Google GenAI")
if prov_ollama: selected_list.append("Ollama (Local)")

plan = checker.evaluate_configuration(selected_list, prov_langsmith)

st.write("### Recommended Production Install Command")
st.code(plan["recommended_install_cmd"], language="bash")

st.write("### Production `.env` Template")
env_lines = [f"{var}=your_api_key_here" for var in plan["required_env_vars"]]
st.code("\n".join(env_lines), language="bash")

# --- SECTION 14: TECHNICAL INTERVIEW PREPARATION ---
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why did LangChain decouple its monolithic package into langchain-core and partner packages?",
     "To resolve dependency conflicts, reduce container image footprints, and allow independent release cycles. Installing langchain-core gives you the Runnable protocol without pulling hundreds of third-party libraries."),
    ("How does LangChain read API keys without passing them explicitly in code?",
     "LangChain classes read default environment variables from os.environ (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY) during initialization. If present, the client configures authentication automatically."),
    ("What is the risk of using python-dotenv in containerized production deployments?",
     "python-dotenv is intended for local development. In production Docker or Kubernetes environments, committing or mounting static .env files risks credential leaks. Secrets should be injected dynamically into container environment variables via Kubernetes Secrets or cloud secret managers."),
    ("What does init_chat_model() do behind the scenes?",
     "It is a universal factory function that dynamically imports the appropriate partner package (e.g. langchain-openai for 'openai'), validates credentials, and instantiates the matching BaseChatModel subclass with standard parameters."),
    ("How do you enable LangSmith tracing without modifying any application code?",
     "By setting two standard environment variables: LANGCHAIN_TRACING_V2='true' and LANGCHAIN_API_KEY='your_key'. LangChain's internal callback manager automatically detects these variables and begins asynchronously streaming execution spans.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Committing `.env` to Git. Accidentally committing live API keys to public or corporate repositories causes automated revocation and security breaches. Always verify `.gitignore`.")
    st.error("**Trap 2 (Breaking Error)**: Monolithic `pip install langchain`. Installing the legacy top-level meta-package installs dozens of unnecessary dependencies and can pin outdated partner package versions.")
with trap_c2:
    st.warning("**Trap 3 (Configuration Warning)**: Forgetting `load_dotenv()`. If running outside Docker, forgetting to invoke `load_dotenv()` before importing LangChain results in missing API key exceptions.")
    st.warning("**Trap 4 (Portability Warning)**: Incompatible Python Version. Attempting to install LangChain v0.3+ on Python 3.8 fails due to Pydantic v2 type requirement incompatibilities. Always target Python 3.10+.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Core Minimal Package"):
        st.markdown("**`langchain-core` contains interfaces, Runnables, and message schemas with zero heavy dependencies.**")
    with st.expander("🎴 Flashcard 2: Standard Provider Packages"):
        st.markdown("**`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Universal Factory"):
        st.markdown("**`init_chat_model(model, model_provider)` dynamically initializes any configured provider model.**")
    with st.expander("🎴 Flashcard 4: LangSmith Activation Env"):
        st.markdown("**`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=...`**")

st.divider()

# --- SECTION 15: SELF-ASSESSMENT CHECKLIST ---
st.header("15. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why LangChain decoupled into core and partner packages.", value=True)
chk2 = st.checkbox("I have configured an isolated Python virtual environment (3.10+).", value=True)
chk3 = st.checkbox("I know how to install and authenticate OpenAI, Anthropic, and Google integrations.", value=True)
chk4 = st.checkbox("I understand how to protect credentials using .env and .gitignore.", value=True)
chk5 = st.checkbox("I can run a programmatic smoke test with init_chat_model to verify connectivity.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 1.3 Installing LangChain. Chapter 1 is fully complete!")
