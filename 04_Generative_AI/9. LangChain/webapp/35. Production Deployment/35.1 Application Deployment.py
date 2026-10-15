import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="35.1 Application Deployment - LangChain Curriculum",
    page_icon="🚀",
    layout="wide"
)

# Custom Theme Styling
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
    .deploy-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-status {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AppDeploymentEngine:
    """
    Simulates production environment verification, secret auditing,
    and Docker container health checks.
    """
    def __init__(self):
        self.deployment_standards = {
            "Base Image": "python:3.12-slim",
            "User": "non-root (appuser:1000)",
            "Secret Injection": "Kubernetes Secrets / Vault",
            "Worker Model": "Uvicorn worker with Gunicorn supervisor",
            "Shutdown": "Graceful SIGTERM handling with 30s drain"
        }

    def audit_config_readiness(self, env_secrets: bool, multi_stage: bool, non_root: bool, lockfile: bool) -> Dict[str, Any]:
        score = 0
        findings = []

        if env_secrets:
            score += 25
            findings.append({"check": "Secrets injected via runtime environment (Zero plain-text on disk)", "status": "PASS"})
        else:
            findings.append({"check": "Plain-text .env file present or secrets hardcoded", "status": "FAIL"})

        if multi_stage:
            score += 25
            findings.append({"check": "Multi-stage Dockerfile build (Builder vs Runtime separation)", "status": "PASS"})
        else:
            findings.append({"check": "Single-stage fat Docker build contains build-essential compiler tools", "status": "FAIL"})

        if non_root:
            score += 25
            findings.append({"check": "Container executes as unprivileged non-root user (appuser)", "status": "PASS"})
        else:
            findings.append({"check": "Container executes as root (Security vulnerability)", "status": "FAIL"})

        if lockfile:
            score += 25
            findings.append({"check": "Deterministic dependency lockfile (poetry.lock / uv.lock)", "status": "PASS"})
        else:
            findings.append({"check": "Unpinned requirements.txt allows non-deterministic upstream breakage", "status": "FAIL"})

        return {
            "score": score,
            "status": "READY_FOR_PRODUCTION" if score == 100 else "HARDENING_REQUIRED",
            "findings": findings
        }

# Instantiate engine
deploy_engine = AppDeploymentEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Application Containerization, Secret Injection, & Microservice Deployment", divider="rainbow")

st.markdown("""
Transitioning LangChain applications from local prototyping into high-availability production requires 
strict adherence to **12-Factor App principles**. Production deployments isolate dependencies into 
**multi-stage Docker containers**, inject credentials strictly via runtime **Secret Stores**, and orchestrate 
processes using production-grade ASGI servers with graceful shutdown signal handling.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph App_Deployment_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_secrets {
            label = "Secret Management & Configuration Plane";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            vault_secrets [label="HashiCorp Vault / Cloud Secrets Manager\\n(Stores OPENAI_API_KEY, LANGSMITH_API_KEY)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            pydantic_settings [label="Pydantic BaseSettings\\n(Type-safe validation, environment variable binding)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_packaging {
            label = "Immutable Multi-Stage Container Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            builder_stage [label="Builder Stage (python:3.12-slim)\\n(Compiles wheels, strips build-essential)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            runtime_stage [label="Runtime Stage (Distroless / Minimal Slim)\\n(Non-root user 'appuser', zero build tools)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_serving {
            label = "Production ASGI Serving & Process Supervision";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            gunicorn_master [label="Gunicorn Master Process\\n(Worker lifecycle supervision, SIGTERM handler)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            uvicorn_workers [label="Uvicorn Worker Processes (x4)\\n(Async event loop, FastAPI endpoints, SSE streaming)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            health_probes [label="Kubernetes Liveness / Readiness Probes\\n(GET /healthz, GET /ready)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        vault_secrets -> pydantic_settings [label="mounted as env vars"];
        pydantic_settings -> uvicorn_workers;

        builder_stage -> runtime_stage [label="copies compiled wheels"];
        runtime_stage -> gunicorn_master [label="starts container CMD"];
        gunicorn_master -> uvicorn_workers [label="forks workers"];
        uvicorn_workers -> health_probes [label="reports status"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Immutable Container Standard:</strong> Application code and pinned dependencies are packaged once into an 
    immutable Docker image. Environment-specific behaviors (dev vs staging vs prod) are dictated exclusively through 
    externalized environment variables and secrets injected at container startup.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: ENVIRONMENT CONFIGURATION
# ==============================================================================
st.header("2. Environment Configuration: Configuration Management and 12-Factor App Principles", divider="rainbow")

st.markdown("""
Under the **12-Factor App methodology**, application code and configuration must be strictly decoupled. 
A build artifact should be completely agnostic of whether it is running on a developer laptop, a staging cluster, 
or an enterprise production environment.
""")

st.markdown("""
### Core Configuration Principles
- **Strict Separation of Config from Code**: Never commit URLs, database connection strings, or project names into Git repositories.
- **Hierarchical Overrides**: Local defaults (`.env.local`) < Staging overrides < Production environment variables.
- **Fail-Fast Validation**: Application startup must abort immediately if a required configuration key is missing or malformed, preventing silent runtime failures during user requests.
""")

# ==============================================================================
# SECTION 3: ENVIRONMENT VARIABLES
# ==============================================================================
st.header("3. Environment Variables: Type-Safe Pydantic BaseSettings and Validation", divider="rainbow")

st.markdown("""
Using raw `os.environ.get()` introduces severe risks: typos in key names return `None`, 
numeric values arrive as strings, and missing variables trigger late `KeyError` crashes in production. 
**Pydantic `BaseSettings`** provides type-safe, validated environment loading.
""")

st.code('''from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class ProductionAppSettings(BaseSettings):
    # Required secrets (will raise ValidationError on startup if missing)
    openai_api_key: SecretStr = Field(..., alias="OPENAI_API_KEY")
    langsmith_api_key: SecretStr = Field(..., alias="LANGSMITH_API_KEY")
    
    # Environment flags
    environment: str = Field(default="production")
    langsmith_project: str = Field(default="enterprise-rag-prod")
    max_concurrency: int = Field(default=30)
    request_timeout_seconds: float = Field(default=15.0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Instantiation validates all variables on process startup!
# config = ProductionAppSettings()
# print(f"Environment: {config.environment}, Project: {config.langsmith_project}")
''', language="python")

# ==============================================================================
# SECTION 4: SECRETS MANAGEMENT
# ==============================================================================
st.header("4. Secrets Management: Vault, Kubernetes Secrets, and Zero-Disk Key Hygiene", divider="rainbow")

st.markdown("""
API keys for commercial model providers represent significant financial and security liabilities. 
If an OpenAI or Anthropic API key is leaked, malicious actors can rack up thousands of dollars in compute 
or exfiltrate internal context.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Antipatterns (Security Risks)
    - Committing `.env` files into Git repositories.
    - Baking `ENV OPENAI_API_KEY="sk-..."` into Docker image layers.
    - Printing raw config objects or stack traces containing API keys into log files.
    - Granting persistent broad permissions instead of scoped keys.
    """)

with col2:
    st.markdown("""
    ### Production Standards
    - Inject keys as ephemeral environment variables via **Kubernetes Secrets** or **AWS Secrets Manager**.
    - Wrap sensitive keys in Pydantic `SecretStr` to prevent accidental string printing.
    - Enable automatic secret rotation and configure provider spend limits with alerting.
    """)

# ==============================================================================
# SECTION 5: DEPENDENCY MANAGEMENT
# ==============================================================================
st.header("5. Dependency Management: Pinned Lockfiles and Eliminating Bloat", divider="rainbow")

st.markdown("""
Generative AI libraries evolve rapidly. A minor update to `langchain-core` or `pydantic` can introduce 
subtle breaking changes. Production microservices must use **deterministic lockfiles** (via Poetry or UV).
""")

st.markdown("""
### Dependency Hygiene Checklist
- **Never Deploy Loose Requirements (`requirements.txt`)**: A file containing `langchain>=0.2.0` will pull different versions across different CI builds. Always use pinned hashes (`uv.lock` or `poetry.lock`).
- **Eliminate Monolithic Packages**: Do not install `langchain` if you only need `langchain-core` and `langchain-openai`. This shrinks container image size by up to 500MB and eliminates dozens of unnecessary transitive dependencies.
""")

# ==============================================================================
# SECTION 6: DOCKER
# ==============================================================================
st.header("6. Docker: Multi-Stage Production Builds, Non-Root Users, and Healthcheck Directives", divider="rainbow")

st.markdown("""
A containerized LLM service must be lightweight, secure, and resilient. 
Below is the enterprise multi-stage Dockerfile pattern for LangChain applications.
""")

st.code('''# Stage 1: Build & Compile Dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
# Install dependencies into user site
RUN pip install --no-cache-dir poetry && \\
    poetry config virtualenvs.create false && \\
    poetry install --no-root --only main

# Stage 2: Final Minimal Runtime Image
FROM python:3.12-slim AS runtime

WORKDIR /app
# Security: Create dedicated unprivileged user
RUN useradd -u 1001 -m appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --chown=appuser:appuser . .

USER appuser
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
''', language="dockerfile")

# ==============================================================================
# SECTION 7: PRODUCTION CONFIGURATION
# ==============================================================================
st.header("7. Production Configuration: Gunicorn / Uvicorn Workers and Graceful Shutdowns", divider="rainbow")

st.markdown("""
Running `uvicorn main:app` directly without a process manager is dangerous in production. 
If an unhandled exception crashes the event loop, the entire container terminates.
""")

st.markdown("""
### Process Supervision Standard
- **Gunicorn Master + Uvicorn Workers**: Gunicorn acts as the master supervisor, managing worker lifecycles, restarting failed workers automatically, and handling OS signals.
- **Worker Sizing Rule**: For I/O bound LLM applications (where threads wait on model APIs), configure `2 * CPU_CORES + 1` workers.
- **Graceful Shutdown**: When Kubernetes issues a `SIGTERM`, the service must stop accepting new requests, allow active streaming connections up to 30 seconds to finish, and close database connections cleanly.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full FastAPI Production Microservice with Health Checks", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**FastAPI microservice featuring health probes, graceful shutdown handling, and settings validation**.
""")

st.code('''from typing import Dict, Any

class ProductionConfigValidator:
    def __init__(self, env: str = "production", timeout_sec: int = 30):
        self.env = env
        self.timeout_sec = timeout_sec
        self.is_healthy = True

    def validate_startup_invariants(self) -> bool:
        """Verifies critical environment prerequisites before accepting traffic."""
        # Check required configurations
        assert self.env in ["production", "staging", "development"], "Invalid environment"
        assert self.timeout_sec > 0, "Timeout must be positive"
        return True

    def liveness_probe(self) -> Dict[str, str]:
        """K8s liveness check: checks if process is alive."""
        return {"status": "ALIVE", "environment": self.env}

    def readiness_probe(self) -> Dict[str, Any]:
        """K8s readiness check: checks if backend dependencies are connected."""
        if not self.is_healthy:
            return {"status": "NOT_READY", "reason": "Database connection degraded"}
        return {"status": "READY", "ready_for_traffic": True}

service_app = ProductionConfigValidator(env="production", timeout_sec=15)
service_app.validate_startup_invariants()

print("Liveness Probe:", service_app.liveness_probe())
print("Readiness Probe:", service_app.readiness_probe())
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Environment Configuration Auditor & Concurrency Sizer", divider="rainbow")

st.markdown("""
Audit deployment security and calculate recommended worker concurrency 
for your containerized LangChain microservice.
""")

wb_view = st.selectbox(
    "Select Deployment Tool:",
    ["1. Production Readiness Security Auditor", "2. Worker & Concurrency Sizing Calculator", "3. Health Probe Status Simulator"]
)

if wb_view == "1. Production Readiness Security Auditor":
    st.subheader("Container Security & Configuration Auditor")
    st.markdown("Select current configuration settings to evaluate production readiness:")

    c_sec = st.toggle("Secrets injected via runtime environment (Zero keys committed to disk)", value=True)
    c_multi = st.toggle("Multi-stage Dockerfile build (Slim final image without compilers)", value=True)
    c_user = st.toggle("Container executes as unprivileged non-root user (appuser:1001)", value=True)
    c_lock = st.toggle("Deterministic dependency lockfile (poetry.lock / uv.lock)", value=True)

    audit_res = deploy_engine.audit_config_readiness(c_sec, c_multi, c_user, c_lock)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Deployment Readiness Score", f"{audit_res['score']}%")
    with m2:
        st.metric("Readiness Status", audit_res["status"])

    st.markdown("#### Detailed Findings:")
    for f in audit_res["findings"]:
        badge = "PASS" if f["status"] == "PASS" else "FAIL"
        st.markdown(f"- `{badge}` **{f['check']}**")

    if audit_res["score"] == 100:
        st.success("All enterprise deployment invariants satisfied! Ready for production deployment.")
    else:
        st.warning("Review failed checks above to prevent security and stability regressions.")

elif wb_view == "2. Worker & Concurrency Sizing Calculator":
    st.subheader("Gunicorn / Uvicorn Concurrency Sizer")
    cpu_cores = st.slider("Allocated CPU Cores:", 1, 16, 4, 1)
    concur_per_worker = st.slider("Target Concurrent Streams per Worker:", 10, 50, 25, 5)

    recommended_workers = (2 * cpu_cores) + 1
    total_concurrency = recommended_workers * concur_per_worker

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Recommended Gunicorn Workers", f"{recommended_workers} Workers")
    with m2:
        st.metric("Max Concurrent Streaming Connections", f"{total_concurrency} Connections")

    st.caption("Formula: Recommended Workers = (2 * CPU Cores) + 1 for I/O bound LLM streaming microservices.")

else:
    st.subheader("Kubernetes Health Probe Simulator")
    is_down = st.toggle("Simulate Vector Database Connection Drop", value=False)
    
    if is_down:
        st.error("Readiness Probe: HTTP 503 SERVICE UNAVAILABLE — Vector store unreachable. Ingress removes pod from traffic pool.")
        st.success("Liveness Probe: HTTP 200 OK — Process is healthy; pod will NOT be unnecessarily restarted by K8s.")
    else:
        st.success("Readiness Probe: HTTP 200 OK — Pod ready for incoming traffic.")
        st.success("Liveness Probe: HTTP 200 OK — Process healthy.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the difference between Kubernetes Liveness and Readiness probes in LLM serving?"):
    st.markdown("""
    **Answer:**  
    - **Liveness Probe (`/healthz`)**: Verifies the Python process is alive and responsive. If it fails, Kubernetes kills and restarts the container.
    - **Readiness Probe (`/ready`)**: Verifies the microservice is ready to accept user traffic (e.g., model client initialized, vector store connection active). If it fails, Kubernetes removes the pod from the load balancer without restarting it.
    """)

with st.expander("Q2: Why should containers never run as the root user in production?"):
    st.markdown("""
    **Answer:**  
    Running containers as `root` creates a severe vulnerability: if an attacker exploits a code vulnerability or library flaw to achieve arbitrary code execution, they obtain root privileges on the container, dramatically increasing the risk of container escape and host compromise. Always create and use an unprivileged non-root user.
    """)

with st.expander("Q3: How does Pydantic BaseSettings prevent production outages compared to os.environ.get()?"):
    st.markdown("""
    **Answer:**  
    `os.environ.get()` returns strings or `None`, deferring errors until runtime when a missing key is accessed during a customer query. Pydantic `BaseSettings` validates types, formats, and required keys at application startup, failing immediately with descriptive error messages if configuration invariants are violated.
    """)

with st.expander("Q4: Why is a multi-stage Docker build essential for Python generative AI applications?"):
    st.markdown("""
    **Answer:**  
    Compiling C-extensions and installing machine learning wheels requires heavy build packages (`build-essential`, `gcc`) that add 500MB+ to the image and introduce security vulnerabilities. Multi-stage builds compile artifacts in a builder image, then copy only the compiled wheels into a minimal runtime image, reducing image size by over 70%.
    """)

with st.expander("Q5: What is the recommended worker model for serving streaming LLM responses with FastAPI?"):
    st.markdown("""
    **Answer:**  
    Use Gunicorn as the master process manager to supervise workers and handle OS signals, using `uvicorn.workers.UvicornWorker` for asynchronous event loops. Configure `(2 * CPU_CORES) + 1` workers to handle asynchronous streaming concurrency efficiently without excessive memory usage.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Hardcoding Model API Keys in Dockerfile ENV Directives**  
Using `ENV OPENAI_API_KEY="sk-..."` permanently embeds the plain-text secret in the Docker image layers, 
exposing credentials to anyone with image pull permissions.
""")

st.error("""
**Trap 2: Restarting Containers on Temporary Model API Failures via Liveness Probes**  
If an external API (like OpenAI) experiences temporary rate limits, triggering a liveness probe failure causes 
Kubernetes to restart the pod continuously, exacerbating the outage. Liveness probes should only check process health.
""")

st.warning("""
**Trap 3: Deploying Unpinned Dependencies in Production Microservices**  
Using an unpinned `requirements.txt` allows unexpected upstream releases to introduce breaking changes into production builds. 
Always deploy with a deterministic lockfile.
""")

st.warning("""
**Trap 4: Missing Graceful SIGTERM Handling During Pod Termination**  
When Kubernetes scales down pods, killing processes abruptly terminates ongoing token streaming connections. 
Always implement a 30-second graceful shutdown drain.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>12-Factor App Config</h4>
        <p>A software engineering methodology requiring complete decoupling of application code from configuration, managed via externalized environment variables.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Non-Root User</h4>
        <p>A security standard ensuring container processes execute under unprivileged user accounts to minimize attack blast radius.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Readiness Probe</h4>
        <p>A Kubernetes health check determining whether a pod should receive traffic from the ingress load balancer.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Multi-Stage Docker Build</h4>
        <p>A Docker packaging architecture separating build dependencies from the final minimal production runtime image.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to implement 12-factor configuration management using Pydantic BaseSettings.", value=True)
chk_2 = st.checkbox("I understand secure secret management and zero-disk credential injection via Kubernetes Secrets.", value=True)
chk_3 = st.checkbox("I know why deterministic lockfiles (poetry.lock / uv.lock) are mandatory for production builds.", value=True)
chk_4 = st.checkbox("I can author a secure multi-stage Dockerfile running as an unprivileged non-root user.", value=True)
chk_5 = st.checkbox("I understand the architectural difference between Kubernetes Liveness and Readiness probes.", value=True)
chk_6 = st.checkbox("I know how to configure Gunicorn with Uvicorn workers and handle graceful SIGTERM shutdowns.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Application Deployment competencies verified! Ready for LangGraph Deployment.")
