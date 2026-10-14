import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="35.3 LangSmith Deployment - LangChain Curriculum",
    page_icon="🛡️",
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
    .ls-badge {
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
class LangSmithDeploymentEngine:
    """
    Simulates enterprise LangSmith deployment management:
    GitHub deployment triggers, autoscaling resource planning,
    monitoring telemetry, and trace diagnostics.
    """
    def __init__(self):
        self.deployment_envs = {
            "Development": {"workers": 1, "concurrency_limit": 10, "tracing_sample_rate": 1.0},
            "Staging": {"workers": 2, "concurrency_limit": 50, "tracing_sample_rate": 1.0},
            "Production": {"workers": 8, "concurrency_limit": 500, "tracing_sample_rate": 0.25}
        }

    def simulate_telemetry_metrics(self, qps: int, env: str = "Production") -> Dict[str, Any]:
        info = self.deployment_envs.get(env, self.deployment_envs["Production"])
        # Simulated metrics based on load
        p95_latency = 380 + int(qps * 4.2)
        error_rate = 0.05 if qps < 80 else 1.8
        tokens_per_min = qps * 60 * 420
        daily_cost_est = round((tokens_per_min * 60 * 24 / 1_000_000) * 0.45, 2)
        
        return {
            "env": env,
            "qps": qps,
            "p95_latency_ms": p95_latency,
            "error_rate_pct": error_rate,
            "tokens_per_min": tokens_per_min,
            "estimated_daily_cost": daily_cost_est,
            "worker_utilization_pct": min(int((qps / info["concurrency_limit"]) * 100), 100)
        }

# Instantiate engine
ls_deploy_engine = LangSmithDeploymentEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise LangSmith Deployment, Cloud Control Plane, & Distributed Agent Telemetry", divider="rainbow")

st.markdown("""
Deploying AI agents to enterprise production requires dedicated **managed agent infrastructure**. 
LangSmith provides an end-to-end deployment control plane that directly connects to GitHub repositories, 
packages LangGraph state machines into high-availability container clusters, exposes secure REST/WebSocket APIs, 
and captures live operational telemetry across every agent turn.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangSmith_Deployment_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_git {
            label = "Continuous Deployment (GitOps)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            git_repo [label="GitHub Repository\\n(langgraph.json + src/agent.py)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            git_webhook [label="GitHub Webhook Trigger\\n(Automatic deploy on git push main)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_control_plane {
            label = "LangSmith Managed Agent Infrastructure";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            image_builder [label="Automated Container Builder\\n(Compiles pinned wheels into minimal runtime)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            agent_cluster [label="Horizontally Scaled Agent Cluster\\n(Auto-scales based on active streaming connections)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            durable_db [label="Managed PostgreSQL & Redis Storage\\n(Thread checkpointers, assistant configurations)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_observability {
            label = "Production Observability & Live Debugging Plane";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            telemetry_bus [label="Real-Time Telemetry Stream\\n(Async trace exporter captures inputs, outputs, errors)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            monitor_dashboard [label="LangSmith Production Dashboard\\n(p95 Latency, Error Rates, Token/Cost Meters)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        git_repo -> git_webhook;
        git_webhook -> image_builder;
        image_builder -> agent_cluster [label="zero-downtime rolling update"];

        agent_cluster -> durable_db [label="state checkpointing"];
        agent_cluster -> telemetry_bus [label="non-blocking trace logs"];
        telemetry_bus -> monitor_dashboard;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Fully Managed Agent Deployment:</strong> LangSmith Deployment abstracts Kubernetes cluster configuration, 
    database connection pooling, and ingress load balancing. Pushing a commit to GitHub automatically triggers a zero-downtime 
    rolling update with built-in rollback capabilities if health probes detect anomalies.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: GITHUB INTEGRATION
# ==============================================================================
st.header("2. GitHub Integration: Continuous Deployment Triggers and Pull Request Previews", divider="rainbow")

st.markdown("""
Modern LLMOps treats agent code as standard version-controlled software. 
LangSmith integrates natively with GitHub organizations to automate deployment pipelines:
""")

st.markdown("""
### GitOps Deployment Capabilities
- **Branch-to-Environment Mapping**: Automatically map `main` branch to Production, `staging` branch to Staging, and feature branches to ephemeral Preview Deployments.
- **PR Preview Environments**: Every Pull Request spins up an isolated sandbox URL where product managers and evaluators can interact with the modified graph before merging.
- **Trace-to-Commit Provenance**: Every trace recorded in LangSmith includes metadata linking back to the exact Git commit SHA that executed the run.
""")

# ==============================================================================
# SECTION 3: DEPLOYMENT CONFIGURATION
# ==============================================================================
st.header("3. Deployment Configuration: Resource Quotas, VPC Peering, and Autoscaling Profiles", divider="rainbow")

st.markdown("""
Deploying agents requires defining operational resource envelopes:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Compute & Network Configuration
    - **Worker Replicas**: Minimum and maximum pod count for Horizontal Pod Autoscaling (HPA).
    - **VPC Peering**: Connect LangSmith agents directly into private corporate AWS/GCP VPCs to access internal databases securely.
    - **Environment Secrets**: Securely inject API keys (`OPENAI_API_KEY`, database credentials) at runtime without baking into images.
    """)

with col2:
    st.markdown("""
    ### Concurrency & Timeout Profiles
    - **Max Connection Concurrency**: Sets the maximum number of simultaneous SSE streaming sessions per worker.
    - **Step Timeout**: Hard cutoff (e.g., 60 seconds) preventing rogue tool calls from hanging indefinitely.
    """)

# ==============================================================================
# SECTION 4: DEPLOYING AGENTS
# ==============================================================================
st.header("4. Deploying Agents: Packaging LangGraph Graphs to Managed Cloud Infrastructure", divider="rainbow")

st.markdown("""
Deploying a LangGraph agent through LangSmith requires a valid `langgraph.json` configuration file 
and compiled graph entrypoint:
""")

st.code('''# Deploy via LangSmith Cloud CLI
# 1. Authenticate with LangSmith
langsmith login

# 2. Deploy current repository
langsmith deploy --project "customer-support-agent" --env "production"
''', language="bash")

st.markdown("""
Once deployed, the agent receives a production endpoint:  
`https://customer-support-agent.langgraph.cloud/v1`
""")

# ==============================================================================
# SECTION 5: PRODUCTION APIS
# ==============================================================================
st.header("5. Production APIs: REST and WebSocket Endpoints and Thread Management", divider="rainbow")

st.markdown("""
LangSmith automatically generates enterprise-grade REST and streaming endpoints 
for deployed agents:
""")

api_df = pd.DataFrame([
    {"Endpoint": "POST /threads", "Role": "Creates a new conversation thread session with persistent state."},
    {"Endpoint": "POST /threads/{id}/runs/stream", "Role": "Dispatches an execution run and streams tokens/deltas via SSE."},
    {"Endpoint": "GET /threads/{id}/state", "Role": "Fetches current snapshot of thread state and history."},
    {"Endpoint": "POST /threads/{id}/state", "Role": "Human-in-the-loop state mutation: updates state before resuming interrupted runs."}
])
st.dataframe(api_df, use_container_width=True)

# ==============================================================================
# SECTION 6: SCALING
# ==============================================================================
st.header("6. Scaling: Distributed Task Workers and Concurrency Limits", divider="rainbow")

st.markdown("""
LLM workloads differ fundamentally from traditional web servers: an individual agent turn may take 
5 to 30 seconds to complete due to multi-tool execution and external model latency.
""")

st.markdown("""
### Scaling Architecture Standards
- **Asynchronous Task Queue**: Requests are enqueued into a distributed queue (Redis / RabbitMQ). Workers pull tasks asynchronously, preventing connection drops during traffic spikes.
- **Connection-Based Autoscaling**: Autoscale agent workers based on active concurrent streaming connections rather than generic CPU or memory utilization.
""")

# ==============================================================================
# SECTION 7: MONITORING
# ==============================================================================
st.header("7. Monitoring: Latency Percentiles (p50/p95/p99) and Token/Cost Gauges", divider="rainbow")

st.markdown("""
Real-time observability is essential for production SLAs. LangSmith dashboards provide 
instant visibility into key Service Level Indicators (SLIs):
""")

st.markdown("""
1. **Time to First Token (TTFT)**: p50 and p95 distributions measuring time before streaming begins.
2. **Step Duration Percentiles**: Breakdown of latency across individual tool calls vs LLM reasoning passes.
3. **Cumulative Token Spend**: Real-time dollar burn rate attributed to model provider and tenant ID.
4. **Error Rate Gauge**: Percentage of runs encountering HTTP 429 rate limits, network timeouts, or schema failures.
""")

# ==============================================================================
# SECTION 8: LOGS
# ==============================================================================
st.header("8. Logs: Structured Event Streaming and Step-Level Execution Logs", divider="rainbow")

st.markdown("""
LangSmith combines application stdout/stderr logs with structured trace spans. 
Every console log statement is synchronized with its corresponding node execution step.
""")

st.code('''import logging
from langsmith import traceable

logger = logging.getLogger("agent.tools")

@traceable(name="execute_database_query")
def execute_database_query(query: str):
    logger.info(f"Dispatching SQL query: {query}")
    # Traced span captures both the log line and the return payload
    return {"rows": 42}
''', language="python")

# ==============================================================================
# SECTION 9: PRODUCTION DEBUGGING
# ==============================================================================
st.header("9. Production Debugging: Interactive Trace Playback and Root-Cause Diagnostics", divider="rainbow")

st.markdown("""
When a production agent encounters an unexpected error or hallucinates an incorrect output, 
engineers use LangSmith's **interactive trace playback** to perform root-cause analysis:
""")

st.markdown("""
- **Step-by-Step State Inspection**: View the exact state dictionary before and after each node executed.
- **Prompt & Response Replay**: View the unvarnished raw prompt text sent to the LLM, including all few-shot examples and system directives.
- **One-Click Playground Export**: Click 'Open in Playground' on any failed trace to tweak system prompts or model parameters and re-run immediately.
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Traced Agent Deployment Script with Health Probes", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**traced agent microservice with thread persistence, health probes, and telemetry emission**.
""")

st.code('''from typing import Dict, Any, List

class DeployedAgentService:
    def __init__(self, service_name: str = "customer-support-agent"):
        self.service_name = service_name
        self.threads_store: Dict[str, List[dict]] = {}
        self.metrics = {"total_runs": 0, "errors": 0}

    def health_check(self) -> Dict[str, str]:
        """Kubernetes liveness and readiness probe."""
        return {"status": "HEALTHY", "service": self.service_name}

    def create_thread(self, thread_id: str) -> Dict[str, str]:
        self.threads_store[thread_id] = []
        return {"thread_id": thread_id, "status": "CREATED"}

    def run_agent_turn(self, thread_id: str, prompt: str) -> Dict[str, Any]:
        """Simulates traced agent execution with state checkpointing."""
        self.metrics["total_runs"] += 1
        history = self.threads_store.setdefault(thread_id, [])

        # Step 1: User message
        history.append({"role": "user", "content": prompt})

        # Step 2: Agent execution & tool reasoning
        simulated_response = f"Agent response for thread {thread_id}: {prompt}"
        history.append({"role": "assistant", "content": simulated_response})

        return {
            "thread_id": thread_id,
            "status": "COMPLETED",
            "reply": simulated_response,
            "turn_count": len(history) // 2
        }

agent_svc = DeployedAgentService()
print("Health Probe Response:", agent_svc.health_check())

# Create thread and run interaction
agent_svc.create_thread("th_session_9901")
res = agent_svc.run_agent_turn("th_session_9901", "How do I check my deployment logs?")
print("Run Turn Output:", res)
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Deployment Scaling Calculator & Trace Diagnostic Sandbox", divider="rainbow")

st.markdown("""
Simulate production traffic loads to estimate worker utilization, p95 latencies, 
and inspect mock trace execution trees.
""")

wb_view = st.selectbox(
    "Select Workbench Mode:",
    ["1. Production Load & Telemetry Simulator", "2. Trace Diagnostic Tree Inspector", "3. Environment Sizing Configurator"]
)

if wb_view == "1. Production Load & Telemetry Simulator":
    st.subheader("Real-Time Load & Telemetry Simulator")
    env_choice = st.selectbox("Select Target Deployment:", ["Production", "Staging", "Development"])
    traffic_qps = st.slider("Incoming Traffic (Queries Per Second):", 5, 200, 35, 5)

    telemetry = ls_deploy_engine.simulate_telemetry_metrics(traffic_qps, env_choice)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("p95 Latency", f"{telemetry['p95_latency_ms']}ms")
    with m2:
        st.metric("Error Rate", f"{telemetry['error_rate_pct']}%")
    with m3:
        st.metric("Worker Pool Load", f"{telemetry['worker_utilization_pct']}%")

    st.info(f"**Tokens Generated per Minute:** `{telemetry['tokens_per_min']:,}` | **Est. Daily Spend:** `${telemetry['estimated_daily_cost']}`")

elif wb_view == "2. Trace Diagnostic Tree Inspector":
    st.subheader("Mock Trace Execution Diagnostic Tree")
    st.markdown("Inspect simulated execution hierarchy for a multi-tool agent turn:")
    st.info("""
    Root Run: `customer_support_graph` (Duration: 840ms, Status: SUCCESS)
    ├── Step 1: `agent_reasoner` (LLM Call: 240ms, Tokens: 420, Cost: $0.001)
    │   └── ToolCall: `fetch_user_billing` (Args: {user_id: 'usr_101'})
    ├── Step 2: `tool_node` (API Call: 180ms, Status: SUCCESS)
    │   └── Observation: `{balance: 45.00, tier: 'enterprise'}`
    └── Step 3: `agent_synthesizer` (LLM Call: 420ms, Tokens: 380, Cost: $0.0009)
        └── Output: "Your current balance is $45.00."
    """)

else:
    st.subheader("Environment Sizing Configurator")
    st.dataframe(pd.DataFrame(ls_deploy_engine.deployment_envs).T, use_container_width=True)
    st.caption("Production environments utilize lower trace sampling rates (25%) to balance observability with telemetry overhead.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does LangSmith managed deployment simplify the operational lifecycle of AI agents?"):
    st.markdown("""
    **Answer:**  
    LangSmith managed deployment provides an end-to-end platform tailored for stateful agents: automated GitHub CI/CD, 
    managed container builds, connection-aware horizontal pod autoscaling, persistent PostgreSQL checkpointers, 
    and unified observability. Teams avoid managing bespoke Kubernetes manifests and database pooling logic.
    """)

with st.expander("Q2: Why should production agent autoscaling be based on concurrent connections rather than CPU?"):
    st.markdown("""
    **Answer:**  
    Agent workloads are heavily I/O bound: workers spend the majority of their time waiting on external model API 
    responses or streaming tokens to clients. CPU utilization often remains low even when an ASGI server has exhausted 
    its connection limits. Scaling on active streaming sessions prevents request drops during traffic spikes.
    """)

with st.expander("Q3: How does trace-to-commit provenance facilitate post-incident root-cause analysis?"):
    st.markdown("""
    **Answer:**  
    Every trace logged by LangSmith automatically records the Git commit SHA, branch, and deployment version that executed it. 
    When an error spike occurs, engineers can instantly determine which code change or prompt update introduced the regression, 
    enabling rapid rollback.
    """)

with st.expander("Q4: What is the purpose of sampling rates in high-throughput production tracing?"):
    st.markdown("""
    **Answer:**  
    In systems processing thousands of queries per second, logging 100% of raw prompts and completions can introduce 
    telemetry cost and network overhead. Setting a sampling rate (e.g., 20% of successful traces, 100% of errors) 
    provides statistically valid SLI metrics while keeping telemetry costs low.
    """)

with st.expander("Q5: How does LangSmith's 'Open in Playground' feature accelerate debugging?"):
    st.markdown("""
    **Answer:**  
    When a production agent produces an erroneous answer or fails to select a tool, clicking 'Open in Playground' 
    exports the exact prompt, system instructions, and tool definitions into an interactive sandbox. Developers can 
    adjust temperature, reword prompts, or test alternate model checkpoints to diagnose the bug immediately.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Deploying Without Request and Step Timeout Enforcements**  
An external tool that hangs indefinitely can hold worker streaming connections open forever, exhausting the worker pool. 
Always enforce strict timeouts on every tool call.
""")

st.error("""
**Trap 2: Logging Sensitive PII into Trace Payloads Without Masking**  
Sending customer credit cards or healthcare data directly into cloud telemetry logs violates regulatory compliance. 
Configure boundary masking or deploy self-hosted LangSmith instances within private VPCs.
""")

st.warning("""
**Trap 3: Using 100% Trace Sampling on High-Volume Production Workloads**  
Recording millions of identical trivial queries bloats trace indexes and incurs unnecessary telemetry spend. 
Sample successful runs while retaining 100% of errors.
""")

st.warning("""
**Trap 4: Deploying Agents Without Health and Readiness Probes**  
Failing to expose `/healthz` endpoints prevents Kubernetes and load balancers from detecting failed workers, 
routing user traffic to unresponsive pods.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>GitOps Deployment</h4>
        <p>A continuous deployment practice where Git commits trigger automated builds, tests, and zero-downtime rolling updates.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Trace Sampling</h4>
        <p>A technique logging a percentage of successful execution spans while retaining 100% of errors to optimize telemetry spend.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Time to First Token (TTFT)</h4>
        <p>A core SLI measuring the latency before streaming begins, critical for user-perceived responsiveness.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Interactive Trace Playback</h4>
        <p>A diagnostic capability enabling step-by-step inspection and sandbox re-execution of historical production runs.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to connect GitHub repositories to LangSmith for continuous deployment.", value=True)
chk_2 = st.checkbox("I understand resource quota configuration, VPC peering, and secret injection.", value=True)
chk_3 = st.checkbox("I can deploy LangGraph agent graphs to managed cloud infrastructure using the CLI.", value=True)
chk_4 = st.checkbox("I understand the production REST and streaming APIs for threads and runs.", value=True)
chk_5 = st.checkbox("I know how to monitor p95 latencies, error rates, and token burn rates in real time.", value=True)
chk_6 = st.checkbox("I can debug production trace execution trees and export failed runs to the playground.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangSmith Deployment competencies verified! Chapter 35 fully completed.")
