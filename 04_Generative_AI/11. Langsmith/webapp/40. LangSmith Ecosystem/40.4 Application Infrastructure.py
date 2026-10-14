import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="40.4 Application Infrastructure - LangSmith Ecosystem",
    page_icon="🏗️",
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
    .infra-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-deploy {
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
class ApplicationInfrastructureEngine:
    """
    Simulates production infrastructure provisioning, container health audits,
    and CI/CD deployment pipelines for LangSmith-instrumented LLM systems.
    """
    def __init__(self):
        self.infra_matrix = {
            "FastAPI": {
                "layer": "API Serving Runtime",
                "core_role": "High-throughput asynchronous REST/SSE streaming endpoints with ASGI concurrency.",
                "telemetry_integration": "Asynchronous background task workers and custom trace context middleware."
            },
            "Docker": {
                "layer": "Containerization & Packaging",
                "core_role": "Immutable, reproducible application images isolating Python environments and C-dependencies.",
                "telemetry_integration": "Injects LANGSMITH_TRACING and API keys via runtime environment secrets."
            },
            "Kubernetes": {
                "layer": "Cluster Orchestration & Auto-Scaling",
                "core_role": "Horizontal Pod Autoscaling (HPA) based on request concurrency, GPU scheduling, and ingress routing.",
                "telemetry_integration": "ConfigMaps and Kubernetes Secrets for cluster-wide project scoping."
            },
            "Cloud Platforms": {
                "layer": "Compute & Storage Infrastructure (AWS/GCP/Azure)",
                "core_role": "Managed VPCs, GPU accelerated instances (A100/H100), and IAM role security.",
                "telemetry_integration": "Hybrid connectivity to LangSmith Cloud (SaaS) or VPC-peered self-hosted instances."
            },
            "CI/CD Systems": {
                "layer": "Continuous Integration & Deployment",
                "core_role": "Automated linting, unit testing, docker build pipelines, and PR regression gates.",
                "telemetry_integration": "Executes client.evaluate() to block merges if candidate accuracy drops below baseline."
            }
        }

    def simulate_deployment_check(self, has_env: bool, has_docker: bool, has_ci: bool) -> Dict[str, Any]:
        score = 0
        checks = []
        if has_env:
            score += 35
            checks.append({"item": "LangSmith Secrets Configured", "status": "PASS"})
        else:
            checks.append({"item": "LangSmith Secrets Missing", "status": "FAIL"})

        if has_docker:
            score += 35
            checks.append({"item": "Container Healthcheck & Multi-Stage Build", "status": "PASS"})
        else:
            checks.append({"item": "Unoptimized Dockerfile", "status": "FAIL"})

        if has_ci:
            score += 30
            checks.append({"item": "Automated Regression Test in CI Pipeline", "status": "PASS"})
        else:
            checks.append({"item": "No CI Regression Quality Gate", "status": "FAIL"})

        return {
            "readiness_score": score,
            "status": "PRODUCTION_READY" if score >= 90 else "NEEDS_ATTENTION",
            "audit_checks": checks
        }

# Instantiate engine
infra_engine = ApplicationInfrastructureEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production LLMOps Infrastructure, Deployment Pipelines, & Tracing Control Plane", divider="rainbow")

st.markdown("""
Scaling generative AI applications beyond local developer laptops requires an **enterprise-grade infrastructure stack**. 
An LLM microservice typically runs on **FastAPI**, packages inside **Docker** containers, scales dynamically across **Kubernetes** clusters, 
deploys on enterprise **Cloud Platforms** (AWS, GCP, Azure), and validates code changes through automated **CI/CD** quality gates. 
LangSmith sits as the observability and evaluation control plane governing this entire infrastructure.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Infrastructure_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_dev_ci {
            label = "CI/CD & Pre-Deployment Testing Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            git_commit [label="Developer Git Push / PR\\n(Feature branch changes)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            github_actions [label="GitHub Actions CI Runner\\n(Runs unit tests & client.evaluate())", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            langsmith_eval [label="LangSmith Golden Evaluation\\n(Blocks PR if quality regresses)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_packaging {
            label = "Containerization & Registry Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            docker_build [label="Multi-Stage Dockerfile Build\\n(Slim Python 3.12, non-root user)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            container_registry [label="Cloud Container Registry\\n(AWS ECR / GCP Artifact Registry)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_runtime {
            label = "Production Cloud & Kubernetes Cluster";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            k8s_ingress [label="Kubernetes Ingress & Traefik/Envoy\\n(SSL termination, load balancing)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            fastapi_pod [label="FastAPI Microservice Pods (HPA)\\n(Async streaming endpoints, LangGraph agent)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            cloud_secrets [label="Cloud Secret Manager (AWS/GCP)\\n(Injects LANGSMITH_API_KEY securely)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_observability {
            label = "LangSmith Observability Control Plane";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            langsmith_cloud [label="LangSmith SaaS / Self-Hosted Control Plane\\n(Aggregates live telemetry, SLI alerts, feedback)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        git_commit -> github_actions;
        github_actions -> langsmith_eval [label="triggers benchmark"];
        langsmith_eval -> docker_build [label="approval on pass"];
        docker_build -> container_registry;

        container_registry -> fastapi_pod [label="deploys image"];
        cloud_secrets -> fastapi_pod [label="mounts env secrets"];
        k8s_ingress -> fastapi_pod [label="routes traffic"];

        fastapi_pod -> langsmith_cloud [label="async non-blocking trace stream"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Full Lifecycle Observability:</strong> The infrastructure stack does not merely host the LLM; 
    it creates a closed feedback loop. CI/CD runs evaluate candidate models before code merges, Docker guarantees 
    dependency consistency, Kubernetes scales based on real-time traffic, and LangSmith captures every operational span.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: FASTAPI
# ==============================================================================
st.header("2. FastAPI: Asynchronous Streaming Endpoints, Background Telemetry Tasks, and Request Contexts", divider="rainbow")

st.markdown("""
**FastAPI** is the industry standard framework for serving generative AI applications due to its high-throughput 
ASGI architecture, native asynchronous I/O (`async`/`await`), and first-class support for **Server-Sent Events (SSE)** 
used in real-time token streaming.
""")

st.markdown("""
### FastAPI Best Practices for LLMOps
1. **Asynchronous Streaming (`StreamingResponse`)**: Stream LLM tokens to clients in real-time while LangSmith collects spans in the background without holding up the connection.
2. **Dynamic Trace Metadata Injection**: Extract request headers (e.g., `X-Tenant-ID`, `X-Session-ID`) in FastAPI middleware and inject them directly into the active LangSmith run tree metadata.
3. **Background Tasks (`BackgroundTasks`)**: Offload post-response logging or feedback capture to asynchronous background threads so HTTP response latency remains under 50ms.
""")

st.code('''from fastapi import FastAPI, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

app = FastAPI(title="Production LLM Service")

@traceable(run_type="chain", name="streaming_support_endpoint")
async def generate_response_stream(query: str, tenant_id: str):
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.metadata["tenant_id"] = tenant_id
        run_tree.metadata["framework"] = "fastapi"

    # Simulated token generator
    tokens = ["This ", "is ", "a ", "streamed ", "FastAPI ", "response."]
    for t in tokens:
        yield t

@app.post("/chat")
async def chat_endpoint(query: str, x_tenant_id: str = Header(default="anonymous")):
    return StreamingResponse(
        generate_response_stream(query, x_tenant_id),
        media_type="text/event-stream"
    )
''', language="python")

# ==============================================================================
# SECTION 3: DOCKER
# ==============================================================================
st.header("3. Docker: Containerizing LLM Applications, Slim Base Images, and Environment Secret Hygiene", divider="rainbow")

st.markdown("""
Packaging LLM applications in **Docker** ensures identical execution environments across local developer laptops, 
CI runners, and production Kubernetes clusters.
""")

st.markdown("""
### Multi-Stage Production Dockerfile Standard
A production Dockerfile must follow strict security and efficiency principles:
- **Slim Base Image**: Use `python:3.12-slim` to eliminate unnecessary OS packages and reduce image size to under 250MB.
- **Non-Root User**: Never run containers as `root`; create a dedicated unprivileged user (`appuser`).
- **Never Bake Secrets**: Never copy `.env` files or API keys into the image. Inject `LANGSMITH_API_KEY` at runtime via Kubernetes secrets.
""")

st.code('''# Production Multi-Stage Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Image
FROM python:3.12-slim AS runtime

WORKDIR /app
# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed wheels from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
''', language="dockerfile")

# ==============================================================================
# SECTION 4: KUBERNETES
# ==============================================================================
st.header("4. Kubernetes: Horizontal Pod Autoscaling (HPA), GPU Slicing, and Cluster Ingress Observability", divider="rainbow")

st.markdown("""
**Kubernetes (K8s)** provides the enterprise container orchestration layer required to manage clusters 
of LLM application pods, auto-scale based on incoming traffic spikes, and manage GPU resources.
""")

c_k1, c_k2 = st.columns(2)
with c_k1:
    st.markdown("""
    ### Horizontal Pod Autoscaling (HPA)
    Traditional CPU/memory metrics fail for LLM workloads because latency degrades due to network I/O and token queues. 
    K8s HPA should scale based on:
    - **Active Concurrency**: Number of open SSE streaming connections.
    - **Queue Depth**: Number of pending prompts waiting for model inference.
    """)

with c_k2:
    st.markdown("""
    ### Cluster Secret Management
    Inject LangSmith configuration cluster-wide via Kubernetes Secrets:
    ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: langsmith-credentials
    type: Opaque
    stringData:
      LANGSMITH_TRACING: "true"
      LANGSMITH_API_KEY: "lsv2_pt_..."
      LANGSMITH_PROJECT: "production-k8s"
    ```
    """)

# ==============================================================================
# SECTION 5: CLOUD PLATFORMS
# ==============================================================================
st.header("5. Cloud Platforms: AWS, GCP, Azure, and Hybrid Cloud Deployments with LangSmith", divider="rainbow")

st.markdown("""
Modern generative AI architectures span across **AWS**, **Google Cloud Platform (GCP)**, and **Microsoft Azure**. 
LangSmith supports both SaaS multi-tenant cloud and VPC-isolated self-hosted enterprise deployments.
""")

cloud_df = pd.DataFrame([
    {"Cloud Provider": "Amazon Web Services (AWS)", "Key LLM Services": "Amazon Bedrock, SageMaker JumpStart, EKS (Elastic Kubernetes)", "LangSmith Architecture": "Connects securely via AWS PrivateLink or VPC peering for self-hosted instances"},
    {"Cloud Provider": "Google Cloud (GCP)", "Key LLM Services": "Google Vertex AI (Gemini 1.5), GKE (Google Kubernetes Engine)", "LangSmith Architecture": "Native support for Gemini token telemetry and Cloud Run serverless deployment"},
    {"Cloud Provider": "Microsoft Azure", "Key LLM Services": "Azure OpenAI Service, Azure AI Search, AKS", "LangSmith Architecture": "Full instrumentation of AzureChatOpenAI with region-specific endpoint routing"}
])
st.dataframe(cloud_df, use_container_width=True)

# ==============================================================================
# SECTION 6: CI/CD SYSTEMS
# ==============================================================================
st.header("6. CI/CD Systems: GitHub Actions, Automated Evaluation Gates, and Continuous Deployment", divider="rainbow")

st.markdown("""
The most transformative LLMOps practice is **Automated Regression Quality Gates in CI/CD**. 
By running LangSmith evaluations inside GitHub Actions or GitLab CI, teams block pull requests 
that degrade model accuracy before they ever reach production.
""")

st.code('''# .github/workflows/llm_regression_gate.yml
name: LLM Regression Quality Gate

on:
  pull_request:
    branches: [main]
    paths:
      - 'prompts/**'
      - 'agents/**'

jobs:
  evaluate-candidate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run LangSmith Evaluation Quality Gate
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m tests.test_regression_eval --tolerance 0.02
''', language="yaml")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full FastAPI Production Service with Health Checks and Tracing", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**FastAPI service with health probes, metadata enrichment, and LangSmith telemetry**.
""")

st.code('''import time
from typing import Dict, Any
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

# 1. Initialize LangSmith Client
client = Client()

# 2. Simulated Core Application Logic
@traceable(run_type="chain", name="enterprise_search_service")
def process_user_query(query: str, tenant_id: str) -> Dict[str, Any]:
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.metadata["tenant_id"] = tenant_id
        run_tree.metadata["service"] = "fastapi-backend"
        run_tree.tags.append("production-api")
        
    # Simulates retrieval + generation
    time.sleep(0.08)
    return {
        "query": query,
        "answer": f"Processed answer for tenant {tenant_id}: {query}",
        "timestamp": time.time()
    }

# 3. FastAPI Service Simulator
class ProductionServiceSimulator:
    def __init__(self):
        self.health_status = "HEALTHY"
        
    def health_check(self) -> Dict[str, str]:
        """K8s liveness & readiness probe endpoint."""
        return {"status": self.health_status, "service": "llm-fastapi-service"}

    def handle_request(self, payload: Dict[str, str]) -> Dict[str, Any]:
        tenant = payload.get("tenant_id", "default-tenant")
        query = payload.get("query", "")
        return process_user_query(query, tenant)

service = ProductionServiceSimulator()
print("Health Probe Response:", service.health_check())

# Simulate Request Ingestion
sample_response = service.handle_request({
    "tenant_id": "cust-enterprise-99",
    "query": "What are the SLA guidelines for K8s pod restarts?"
})
print("Service Response:", sample_response)
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Application Infrastructure Configurator & Deployment Readiness Inspector", divider="rainbow")

st.markdown("""
Configure your application infrastructure parameters and run an automated 
production readiness audit across your deployment components.
""")

wb_view = st.selectbox(
    "Select Infrastructure Tool:",
    ["1. Production Readiness Audit", "2. Infrastructure Layer Matrix Inspector", "3. HPA Scaling Rule Calculator"]
)

if wb_view == "1. Production Readiness Audit":
    st.subheader("Enterprise Deployment Readiness Inspector")
    st.markdown("Verify that your LLMOps infrastructure satisfies production security, packaging, and CI/CD criteria:")

    c_env = st.toggle("LangSmith Secrets Configured (Non-hardcoded runtime injection)", value=True)
    c_dock = st.toggle("Multi-Stage Dockerfile with Non-Root User & Slim Base Image", value=True)
    c_ci = st.toggle("Automated LangSmith Regression Testing Gate in CI/CD (GitHub Actions)", value=True)

    audit_res = infra_engine.simulate_deployment_check(c_env, c_dock, c_ci)

    score_val = audit_res["readiness_score"]
    st.metric("Infrastructure Readiness Score", f"{score_val}%")

    if score_val >= 90:
        st.success(f"Status: {audit_res['status']} — All foundational LLMOps infrastructure requirements satisfied!")
    else:
        st.warning(f"Status: {audit_res['status']} — Critical infrastructure gaps detected. Review checklist below.")

    st.markdown("#### Audit Checklist Results:")
    for chk in audit_res["audit_checks"]:
        icon = "PASS" if chk["status"] == "PASS" else "FAIL"
        st.markdown(f"- `{icon}` **{chk['item']}**")

elif wb_view == "2. Infrastructure Layer Matrix Inspector":
    st.subheader("Infrastructure Stack Functional Matrix")
    sel_layer = st.selectbox("Select Layer:", list(infra_engine.infra_matrix.keys()))
    l_info = infra_engine.infra_matrix[sel_layer]

    st.markdown(f"### {sel_layer}")
    st.info(f"**Layer Category:** `{l_info['layer']}`")
    st.markdown(f"**Core Operational Role:** {l_info['core_role']}")
    st.success(f"**LangSmith Telemetry Integration:** {l_info['telemetry_integration']}")

else:
    st.subheader("Kubernetes HPA Auto-Scaling Calculator")
    st.markdown("Estimate required pod replicas based on concurrent user requests and streaming connection limits:")

    concur_per_pod = st.slider("Maximum Concurrency per FastAPI Pod:", 10, 100, 30, 5)
    total_traffic = st.slider("Expected Peak Concurrent Users:", 50, 2000, 300, 50)

    needed_pods = int(np.ceil(total_traffic / concur_per_pod))
    st.metric("Estimated Kubernetes Pod Replicas (HPA Target)", f"{needed_pods} Pods")
    st.caption("Auto-scaling rule: Scale out when active SSE streaming connections exceed 80% of target concurrency.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How should Horizontal Pod Autoscalers (HPA) be configured for LLM serving microservices on Kubernetes?"):
    st.markdown("""
    **Answer:**  
    Standard CPU and memory metrics are poor indicators for LLM microservices because streaming inference 
    is I/O bound. HPAs should scale on custom metrics: active concurrent connections (SSE streams), 
    request queue depth, or Time to First Token (TTFT) degradation. Scaling on concurrency prevents request 
    timeouts during sudden traffic bursts.
    """)

with st.expander("Q2: Why is multi-stage building critical when containerizing Python LLM applications with Docker?"):
    st.markdown("""
    **Answer:**  
    Building packages (such as `numpy`, `tiktoken`, or C-extensions) requires build tools (`gcc`, `g++`, `build-essential`) 
    that bloat image size and introduce CVE security vulnerabilities. A multi-stage build uses a temporary builder image 
    to compile wheels, then copies only the compiled runtime files into a minimal `python-slim` final image, 
    reducing size by over 70% and minimizing the attack surface.
    """)

with st.expander("Q3: How do you prevent sensitive API keys from leaking into Docker images or public Git repositories?"):
    st.markdown("""
    **Answer:**  
    1. Add `.env`, `*.key`, and `secrets/` to `.gitignore` and `.dockerignore`.
    2. Never use `ENV` instructions in Dockerfiles for secrets.
    3. Mount secrets at runtime using Kubernetes Secrets, AWS Secrets Manager, or HashiCorp Vault.
    4. Implement pre-commit hooks (e.g., `git-secrets`, `trufflehog`) to block accidental commits.
    """)

with st.expander("Q4: How do automated CI/CD regression gates protect production generative AI applications?"):
    st.markdown("""
    **Answer:**  
    When a developer submits a PR altering prompts, model versions, or chunking parameters, GitHub Actions executes 
    a test script that runs `client.evaluate()` against a versioned LangSmith Golden Dataset. If candidate scores 
    drop by more than an allowable tolerance (e.g., > 2%), the CI build fails and blocks the PR from merging.
    """)

with st.expander("Q5: What is the recommended strategy for LangSmith deployment in enterprise air-gapped VPCs?"):
    st.markdown("""
    **Answer:**  
    Enterprises with strict data residency regulations deploy **Self-Hosted LangSmith** within their own 
    AWS/GCP/Azure Kubernetes cluster (EKS/GKE). The LangSmith backend runs on internal PostgreSQL and ClickHouse 
    storage, ensuring that zero trace payloads or prompt strings leave the corporate VPC boundary.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Running Containerized FastAPI Services as the Root User**  
Running Docker containers as `root` allows potential container-escape vulnerabilities to compromise the host node. 
Always define and use a non-root `appuser`.
""")

st.error("""
**Trap 2: Baking API Keys and Secrets Directly into Docker Image Layers**  
Executing `ENV LANGSMITH_API_KEY="lsv2_pt_..."` inside a Dockerfile bakes the plain-text key into image history. 
Anyone with image pull access can extract your credentials.
""")

st.warning("""
**Trap 3: Using CPU-Based Kubernetes Autoscaling for Streaming I/O Workloads**  
Pods serving streaming responses can become completely bottlenecked by connection limits while CPU utilization 
remains under 30%, causing requests to drop without triggering an HPA scale-out.
""")

st.warning("""
**Trap 4: Deploying Prompt Changes Directly to Production Without CI Regression Evaluation**  
Modifying a prompt in production based on a single successful test query almost always causes silent regressions 
on edge cases. Never deploy without automated CI/CD evaluation runs.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>ASGI Concurrency</h4>
        <p>Asynchronous Server Gateway Interface enabling FastAPI to handle thousands of concurrent streaming connections efficiently.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Multi-Stage Build</h4>
        <p>Docker technique isolating compilation tools into a builder stage to produce minimal, secure production runtime images.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Concurrency-Based HPA</h4>
        <p>Kubernetes autoscaling based on active streaming connections and queue depth rather than generic CPU usage.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>CI Regression Gate</h4>
        <p>An automated pull-request validation step running LangSmith evaluations to block code that degrades benchmark accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to design asynchronous FastAPI streaming endpoints with LangSmith metadata injection.", value=True)
chk_2 = st.checkbox("I understand multi-stage Docker builds, slim base images, and non-root security standards.", value=True)
chk_3 = st.checkbox("I can configure Kubernetes HPA auto-scaling based on streaming concurrency rather than CPU.", value=True)
chk_4 = st.checkbox("I understand cloud deployment architectures across AWS, GCP, Azure, and self-hosted LangSmith VPCs.", value=True)
chk_5 = st.checkbox("I know how to build automated GitHub Actions CI/CD workflows executing LangSmith regression gates.", value=True)
chk_6 = st.checkbox("I can implement secure runtime secret management preventing API key leakage into container layers.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Application Infrastructure ecosystem mastery verified! Chapter 40 fully completed.")
