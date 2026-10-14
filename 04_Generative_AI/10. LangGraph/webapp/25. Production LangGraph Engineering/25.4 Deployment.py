import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="25.4 Deployment - LangGraph Curriculum",
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
    .badge-deploy {
        background-color: #083344;
        color: #22D3EE;
        border: 1px solid #0891B2;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class DeploymentScalingSimulator:
    """
    Simulates Kubernetes pod scaling, memory footprint, and database connection
    pool utilization under varying concurrent LangGraph request loads.
    """
    def __init__(self):
        pass

    def calculate_cluster_requirements(
        self,
        concurrent_users: int,
        requests_per_user_per_min: int,
        avg_turn_latency_sec: float
    ) -> Dict[str, Any]:
        total_rpm = concurrent_users * requests_per_user_per_min
        total_rps = total_rpm / 60.0
        
        # Concurrency capacity per pod (async uvicorn handles ~25 active streaming connections comfortably)
        capacity_per_pod_rps = 15.0
        required_pods = max(int(np.ceil(total_rps / capacity_per_pod_rps)), 2)
        
        # Resource allocation
        cpu_cores_per_pod = 2
        ram_gb_per_pod = 4
        total_cpu = required_pods * cpu_cores_per_pod
        total_ram = required_pods * ram_gb_per_pod
        
        # Postgres connection pool sizing (each pod needs ~10 pool connections for PostgresSaver)
        pool_connections_per_pod = 10
        total_db_connections = required_pods * pool_connections_per_pod
        
        return {
            "total_rps": round(total_rps, 1),
            "required_pods": required_pods,
            "total_cpu_cores": total_cpu,
            "total_ram_gb": total_ram,
            "total_db_connections": total_db_connections,
            "estimated_monthly_compute_usd": required_pods * 72  # Estimated cloud node share
        }

deploy_sim = DeploymentScalingSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Deployment Lifecycles, Containerization, and Cloud Infrastructure Topology", divider="rainbow")

st.markdown("""
Deploying LangGraph into production environments requires a robust DevOps lifecycle: transitioning from local 
rapid-prototyping in LangGraph Studio, to standardized Docker container packaging, to scalable cloud deployments 
on Kubernetes or managed LangGraph Cloud. Because LangGraph agents are stateful, long-running, and streaming-centric, 
deployment infrastructure must be architected for connection durability, graceful termination, and seamless horizontal scaling.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DeploymentBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_dev {
            label="1. Local Development Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            local_cli [label="langgraph dev CLI\\nHot-Reloading Python Runtime", fillcolor="#0F172A"];
            studio_ui [label="LangGraph Studio Desktop UI\\nVisual State Scrubbing & Inspection", fillcolor="#1E293B"];
        }

        subgraph cluster_packaging {
            label="2. Artifact Packaging Tier";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            manifest [label="langgraph.json Manifest\\nDeclares Graphs, Dependencies & Env", fillcolor="#0F172A"];
            docker_img [label="Multi-Stage Docker Image\\nDistroless / Python 3.12-Slim\\nNon-Root Security Hardened", fillcolor="#083344", fontcolor="#67E8F9"];
        }

        subgraph cluster_prod {
            label="3. Cloud Production Cluster (Kubernetes / EKS / GKE)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            ingress [label="Ingress Controller / AWS ALB\\nHandles HTTP/2, SSE Long-Lived Streams, TLS", fillcolor="#1E293B"];
            pod_1 [label="LangGraph Server Pod 1\\nUvicorn / FastAPI Workers", fillcolor="#1E293B"];
            pod_2 [label="LangGraph Server Pod 2\\nUvicorn / FastAPI Workers", fillcolor="#1E293B"];
            pod_n [label="LangGraph Server Pod N (HPA)\\nAutoscales on CPU / Concurrency", fillcolor="#1E293B"];
        }

        subgraph cluster_persistence {
            label="4. Managed Persistence & Gateway";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            pg_cluster [label="Postgres Cluster (RDS Aurora)\\nPostgresSaver Connection Pool", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            redis_cache [label="Redis Elasticache\\nSession Cache & Rate Limiting", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        local_cli -> studio_ui;
        local_cli -> manifest;
        manifest -> docker_img;

        docker_img -> pod_1 [label="K8s Deployment"];
        docker_img -> pod_2;
        docker_img -> pod_n;

        ingress -> pod_1 [label="SSE Streaming Connection"];
        ingress -> pod_2;
        ingress -> pod_n;

        pod_1 -> pg_cluster [label="Checkpoints"];
        pod_2 -> pg_cluster;
        pod_n -> pg_cluster;
        pod_1 -> redis_cache;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Local Deployment
# ==============================================================================
st.header("2. Local Deployment: Development Servers, LangGraph Studio, and Hot-Reload Environments", divider="rainbow")

st.markdown("""
During local development, engineering speed depends on rapid feedback cycles. LangGraph provides dedicated tooling 
to develop, inspect, and debug graphs locally without managing cloud infrastructure:

### Local Development CLI:
- **`langgraph dev`**: Automatically starts a local LangGraph API server with file watchers for hot reloading.
- **LangGraph Studio**: A visual desktop interface that connects to `langgraph dev`. It visualizes the compiled DAG, 
  lets engineers inspect intermediate node outputs, edit state snapshots, and trigger time-travel rollbacks in real time.
""")

st.code('''# Install LangGraph CLI
pip install langgraph-cli

# Start local server with Studio visualizer
langgraph dev --port 8123
''', language="bash")

# ==============================================================================
# SECTION 3: Server Deployment
# ==============================================================================
st.header("3. Server Deployment: Production WSGI/ASGI Hosts, Uvicorn, Gunicorn, and Process Supervisors", divider="rainbow")

st.markdown("""
When deploying custom LangGraph FastAPI services on standard virtual machines (such as EC2 or Compute Engine), 
applications are hosted using production ASGI servers:

- **Uvicorn Workers with Gunicorn Supervisor**: Gunicorn manages worker lifecycles while Uvicorn handles async HTTP/2 
  and SSE streaming event loops.
- **Worker Concurrency**: Set worker count to `(2 x CPU Cores) + 1` for CPU-bound workloads, or utilize async concurrency 
  settings (`--limit-concurrency 1000`) for I/O-bound agent streaming.
- **Keep-Alive Timeouts**: Configure long keep-alive timeouts (e.g., `--timeout-keep-alive 120`) to prevent reverse 
  proxies from terminating active Server-Sent Events (SSE) streaming connections.
""")

# ==============================================================================
# SECTION 4: LangGraph Deployment
# ==============================================================================
st.header("4. LangGraph Deployment: LangGraph Server CLI, langgraph.json Manifests, and Native Cloud Engine", divider="rainbow")

st.markdown("""
LangGraph Server is the official runtime designed specifically for executing LangGraph graphs in production. 
It natively provides persistence, streaming, task queues, and multi-tenant thread management out of the box.

### Configuration Manifest (`langgraph.json`):
Every LangGraph deployment is configured via a `langgraph.json` file in the project root:
""")

st.code('''{
  "python_version": "3.12",
  "dependencies": [
    "."
  ],
  "graphs": {
    "customer_support": "./src/agents/support.py:graph",
    "data_analyst": "./src/agents/analyst.py:analyst_app"
  },
  "env": ".env.production"
}
''', language="json")

st.markdown("""
### Deployment Modes:
1. **Self-Hosted LangGraph Server**: Run via Docker container on your own VPC or Kubernetes cluster.
2. **LangGraph Cloud**: Fully managed SaaS platform hosted by LangChain, offering serverless auto-scaling and zero-ops checkpointer management.
""")

# ==============================================================================
# SECTION 5: Docker
# ==============================================================================
st.header("5. Docker: Multi-Stage Containerization, Production Base Images, and Security Hardening", divider="rainbow")

st.markdown("""
Production containers must be lean, secure, and reproducible. A multi-stage Docker build isolates dependency compilation 
from the final runtime image, minimizing image size and eliminating vulnerability attack surfaces.

### Best Practices for LangGraph Dockerfiles:
- **Python 3.12-Slim Base**: Avoid bulky base images; use slim variants.
- **Non-Root Execution**: Run the application process as a dedicated unprivileged user (`appuser`).
- **Pre-Compiled Wheels**: Compile C-dependencies in a build stage and copy only wheels to the production image.
""")

st.code('''# Multi-stage Dockerfile for LangGraph Application
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY src/ /app/src/
RUN chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
''', language="dockerfile")

# ==============================================================================
# SECTION 6: Cloud Deployment
# ==============================================================================
st.header("6. Cloud Deployment: Kubernetes Deployments, AWS ECS, GCP Cloud Run, and Serverless Containers", divider="rainbow")

st.markdown("""
Depending on your enterprise cloud strategy, LangGraph applications can be deployed across several container runtimes:

| Target Platform | Streaming Support | Max Execution Duration | Ideal Workload |
| :--- | :--- | :--- | :--- |
| **Kubernetes (EKS / GKE)** | Native HTTP/2 & SSE | Unlimited | Enterprise agents, multi-tenant clusters, custom VPCs |
| **AWS ECS Fargate** | Excellent (via ALB) | Unlimited | Managed containers without K8s control plane overhead |
| **GCP Cloud Run** | Supported (via HTTP/2) | 60 Minutes | Scalable event-driven webhooks and bursty user traffic |
| **AWS Lambda** | Limited (Buffered) | 15 Minutes | Short agent steps (Not recommended for long streaming sessions) |
""")

# ==============================================================================
# SECTION 7: Environment Configuration
# ==============================================================================
st.header("7. Environment Configuration: 12-Factor App Principles, .env Management, and Secret Injection", divider="rainbow")

st.markdown("""
In accordance with 12-Factor App methodology, configuration is strictly separated from application code:

- **Strict Environment Injection**: Database URIs, model API keys, and project IDs are injected at runtime via container environment variables.
- **Pydantic BaseSettings**: Validate all required environment variables on startup. If `DATABASE_URL` or `LANGSMITH_API_KEY` 
  is missing, fail loudly during container initialization rather than crashing during a live user request.
""")

st.code('''from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field

class ProductionConfig(BaseSettings):
    database_url: SecretStr = Field(..., env="DATABASE_URL")
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    langsmith_api_key: Optional[SecretStr] = Field(None, env="LANGSMITH_API_KEY")
    environment: str = Field("production", env="APP_ENV")
    max_thread_workers: int = Field(20, env="MAX_THREAD_WORKERS")

    class Config:
        env_file = ".env"

# Instantiation validates all variables immediately
config = ProductionConfig()
''', language="python")

# ==============================================================================
# SECTION 8: Scaling
# ==============================================================================
st.header("8. Scaling: Horizontal Pod Autoscaling (HPA), Concurrency Limits, and Checkpoint Connection Sizing", divider="rainbow")

st.markdown("""
Scaling stateful agents horizontally requires careful sizing of database connection pools and autoscaling triggers:

1. **Autoscaling Metrics**: Do not scale solely on CPU utilization; agent workers frequently wait on model provider I/O 
   where CPU remains low. Scale on **active HTTP connections** or **message queue depth**.
2. **Database Connection Sizing**: Each replica pod maintains a pool of connections for `PostgresSaver`. 
   If your Postgres database allows `max_connections = 200`, and each pod uses a pool of 15 connections, 
   the cluster cannot safely scale beyond 12 pods without exhausting database connections. 
   Deploy **PgBouncer** in transaction-pooling mode to support hundreds of replica pods.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Stage Dockerfile, langgraph.json Manifest, and Kubernetes Spec", divider="rainbow")

st.markdown("""
Below is an enterprise Kubernetes Deployment and Service specification configured for long-lived SSE streaming, 
health probes, and graceful pod termination.
""")

st.code('''apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-agent-deployment
  namespace: production
  labels:
    app: langgraph-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-agent
  template:
    metadata:
      labels:
        app: langgraph-agent
    spec:
      containers:
      - name: agent-runtime
        image: registry.enterprise.internal/agents/support-graph:v2.1.0
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: agent-secrets
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        # Allow active streaming connections up to 60 seconds to finish on redeployment
        terminationGracePeriodSeconds: 60
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-agent-service
  namespace: production
spec:
  selector:
    app: langgraph-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
''', language="yaml")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Deployment Topology Configurator & Horizontal Scaling Estimator", divider="rainbow")

st.markdown("""
Calculate production compute requirements, Kubernetes pod counts, and database connection pool sizing 
based on anticipated concurrent agent workloads.
""")

col_calc1, col_calc2 = st.columns([1, 1])

with col_calc1:
    st.subheader("Workload Sizing Parameters")
    users = st.slider("Concurrent Active Users", min_value=10, max_value=1000, value=150, step=10)
    rpm_per_user = st.slider("Requests per User per Minute", min_value=1, max_value=10, value=3)
    latency = st.slider("Average Turn Latency (Seconds)", min_value=0.5, max_value=10.0, value=2.5, step=0.5)

with col_calc2:
    st.subheader("Cluster Sizing Estimates")
    sizing = deploy_sim.calculate_cluster_requirements(users, rpm_per_user, latency)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Total Ingress Throughput", f"{sizing['total_rps']} RPS")
        st.metric("Required Replica Pods", f"{sizing['required_pods']} Pods")
    with m2:
        st.metric("Total Compute Resources", f"{sizing['total_cpu_cores']} vCPU / {sizing['total_ram_gb']} GB")
        st.metric("Postgres Pool Connections", f"{sizing['total_db_connections']} Conns")
        
    st.info(f"Estimated Monthly Compute Cost: ~{sizing['estimated_monthly_compute_usd']} USD / month (excluding model inference).")
    if sizing["total_db_connections"] > 100:
        st.warning("High Database Connection Count: Deploying PgBouncer in front of PostgresSaver is strongly recommended!")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How do long-running SSE streaming connections affect Kubernetes rolling deployments?"):
    st.markdown("""
    **Answer:**  
    When a deployment updates, Kubernetes sends a `SIGTERM` signal to old pods. If `terminationGracePeriodSeconds` is short (e.g., default 30s), 
    active agent streaming connections are abruptly severed, resulting in broken responses for users. 
    Production deployments must configure graceful termination handlers that reject new connections while allowing active SSE streams to complete.
    """)

with st.expander("Q2: Why is Serverless AWS Lambda generally ill-suited for multi-turn LangGraph agents?"):
    st.markdown("""
    **Answer:**  
    1. **Execution Limits**: Lambda has a hard 15-minute timeout, whereas research agents or HITL workflows may span hours or days.  
    2. **Connection Pooling**: Lambda cold starts and ephemeral lifecycles prevent persistent database connection pooling for `PostgresSaver`.  
    3. **Streaming Support**: True token-by-token SSE streaming over API Gateway and Lambda requires specialized Response Streaming adapters.
    """)

with st.expander("Q3: What role does langgraph.json play in production LangGraph Server deployments?"):
    st.markdown("""
    **Answer:**  
    `langgraph.json` acts as the manifest contract declaring which Python files and compiled graph objects to export, 
    what environment files to load, and which dependencies to install. It enables the LangGraph CLI and Cloud runner 
    to automatically construct endpoints and route requests.
    """)

with st.expander("Q4: How do you prevent database connection pool exhaustion when scaling worker pods horizontally?"):
    st.markdown("""
    **Answer:**  
    Deploy an external connection pooler like **PgBouncer** or **AWS RDS Proxy** between the Kubernetes pods and the PostgreSQL database. 
    Configure the pooler in transaction pooling mode, allowing hundreds of stateless pods to share a small pool of physical database connections.
    """)

with st.expander("Q5: What health check endpoints should be exposed by a production LangGraph microservice?"):
    st.markdown("""
    **Answer:**  
    - `/healthz` (Liveness): Confirms the Uvicorn event loop is responsive.  
    - `/ready` (Readiness): Verifies that database checkpointer connections, Redis queues, and LLM gateway endpoints are healthy before routing ingress traffic.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Default Ingress Timeouts Killing Streaming Connections**  
Standard Nginx Ingress or AWS ALB controllers have default proxy read timeouts of 60 seconds. 
If an agent step or tool call takes longer than 60 seconds without emitting a token, the proxy drops the connection with a 504 Gateway Timeout. 
Always configure ingress timeouts to at least 300 seconds for agent workloads.
""")

st.error("""
**Trap 2: Running Containers as Root User**  
Deploying containers with the default root user allows container escape vulnerabilities to compromise the host node. 
Always enforce a non-root `USER appuser` directive in production Dockerfiles.
""")

st.warning("""
**Trap 3: Hardcoding Graph Objects in Server Entrypoints**  
Compiling graphs with in-memory checkpointers at module import time prevents configuring dynamic per-tenant checkpointers 
or environment variables during server boot.
""")

st.warning("""
**Trap 4: Neglecting Pod Disruption Budgets (PDB)**  
During Kubernetes cluster node upgrades, neglecting to define a PodDisruptionBudget can cause all agent pods to be drained 
simultaneously, causing a complete service outage.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>langgraph.json</h4>
        <p>The manifest file defining compiled graph endpoints, Python runtime versions, and environment configurations for LangGraph Server.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Multi-Stage Docker Build</h4>
        <p>A container build pattern that compiles dependencies in a temporary stage and copies only artifacts to a minimal, secure runtime image.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>terminationGracePeriodSeconds</h4>
        <p>A Kubernetes deployment configuration defining how long terminating pods are given to finish active streaming connections before SIGKILL.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>PgBouncer Connection Pooling</h4>
        <p>A lightweight connection pooler enabling high-concurrency horizontal pod autoscaling without exhausting PostgreSQL connection limits.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to develop and debug graphs locally using langgraph dev and LangGraph Studio.", value=True)
chk_2 = st.checkbox("I can configure langgraph.json manifests to deploy graphs on LangGraph Server.", value=True)
chk_3 = st.checkbox("I understand how to build lean, multi-stage Dockerfiles with non-root security privileges.", value=True)
chk_4 = st.checkbox("I know how to configure Kubernetes deployments with graceful termination for active SSE streams.", value=True)
chk_5 = st.checkbox("I can manage environment variables and secrets following 12-Factor App methodology.", value=True)
chk_6 = st.checkbox("I understand how to scale pods horizontally and size Postgres connection pools with PgBouncer.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Deployment and container orchestration mastery confirmed! Ready for production reliability engineering.")
