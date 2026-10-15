import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="34.3 Continuous Deployment - LangSmith Master Curriculum",
    page_icon="🚀",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LLMDeploymentManagerEngine:
    """
    Enterprise Continuous Deployment & Zero-Downtime Rollback Engine.
    Emulates dynamic Prompt Hub tag switching, Canary traffic splitting,
    StateGraph compiled agent deployment, version lineage auditing, and automated rollback triggers.
    """
    def __init__(self, app_name: str = "enterprise-support-agent"):
        self.app_name = app_name
        self.deployment_history: List[Dict[str, Any]] = []

    def deploy_release(
        self,
        target_version: str,
        prompt_tag: str,
        model_endpoint: str,
        traffic_split_percent: int,
        canary_error_threshold_percent: float = 2.0,
        inject_canary_outage: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a canary deployment stage, simulating health telemetry and automated rollback criteria.
        """
        # Baseline production state
        active_prod_version = "v1.8.0-stable"
        active_prod_prompt_tag = "prod"
        active_prod_endpoint = "gpt-4o-2024-08-06"

        # Telemetry generation
        if inject_canary_outage:
            canary_error_rate = 5.8
            canary_p95_ms = 4120.0
            rollback_triggered = True
            rollback_reason = f"Canary error rate {canary_error_rate:.1f}% breached threshold {canary_error_threshold_percent:.1f}%."
            final_status = "AUTOMATIC_ROLLBACK_EXECUTED"
            active_live_version = active_prod_version
            active_live_prompt_tag = active_prod_prompt_tag
            active_live_endpoint = active_prod_endpoint
            effective_traffic_split = 0
        else:
            canary_error_rate = 0.4
            canary_p95_ms = 1640.0
            rollback_triggered = False
            rollback_reason = "None"
            final_status = "CANARY_HEALTHY" if traffic_split_percent < 100 else "FULLY_PROMOTED_TO_PRODUCTION"
            active_live_version = target_version
            active_live_prompt_tag = prompt_tag
            active_live_endpoint = model_endpoint
            effective_traffic_split = traffic_split_percent

        report = {
            "deployment_id": f"deploy-{target_version.replace('.', '_')}",
            "target_version": target_version,
            "prompt_tag": prompt_tag,
            "model_endpoint": model_endpoint,
            "traffic_split_percent": effective_traffic_split,
            "canary_telemetry": {
                "error_rate_percent": canary_error_rate,
                "p95_latency_ms": canary_p95_ms,
                "sample_requests_evaluated": 1500
            },
            "rollback_triggered": rollback_triggered,
            "rollback_reason": rollback_reason,
            "final_status": final_status,
            "active_live_version": active_live_version,
            "active_live_prompt_tag": active_live_prompt_tag,
            "active_live_endpoint": active_live_endpoint
        }
        self.deployment_history.append(report)
        return report

# Initialize engine in session state
if "deploy_manager_engine" not in st.session_state:
    st.session_state.deploy_manager_engine = LLMDeploymentManagerEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("🚀 Chapter 34.3: Continuous Deployment for LLM Applications")
st.caption("Prompt Hub Tagging, Canary Traffic Routing, Compiled Agent Releases, Version Governance, and Zero-Downtime Rollbacks")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Continuous Deployment and Zero-Downtime Rollback Architecture", divider="rainbow")

st.markdown("""
Continuous Deployment (CD) for LLM applications requires a paradigm shift from traditional stateless microservices. 
In GenAI stacks, a "release" is rarely just binary code; it is a composite artifact of system prompts, model endpoints, 
temperature parameters, tool definitions, retrieval index versions, and stateful agent graph topologies. 
Enterprise CD automates the progressive promotion of these artifacts with zero application downtime and instant rollback guarantees.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ContinuousDeploymentPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_artifacts {
            label = "1. Versioned Release Artifacts";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            A1 [label="LangSmith Prompt Hub\\n(Tag: :staging -> :prod)", fillcolor="#334155", color="#64748B"];
            A2 [label="LLM Gateway Config\\n(Weights: gpt-4o vs claude-3-5)", fillcolor="#334155", color="#64748B"];
            A3 [label="LangGraph Compiled Artifact\\n(State Schema & Checkpointer)", fillcolor="#334155", color="#64748B"];
        }

        subgraph cluster_gateway {
            label = "2. Dynamic Ingress & Traffic Splitter";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            GW [label="AI Gateway / Proxy\\n(LiteLLM / Cloudflare / Envoy)", fillcolor="#1E3A8A", color="#3B82F6"];
            SP_BLUE [label="Stable Baseline (90% Traffic)\\nPrompt: :prod | Model: gpt-4o-2024-08", fillcolor="#047857", color="#10B981"];
            SP_GREEN [label="Canary Release (10% Traffic)\\nPrompt: :canary | Model: gpt-4o-2024-11", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_telemetry {
            label = "3. Real-Time Canary Health Telemetry";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            T1 [label="LangSmith Online Tracing Stream\\n(Latency p95, Error Codes)", fillcolor="#0E7490", color="#06B6D4"];
            T2 [label="Online Evaluation Watcher\\n(Hallucination & Faithfulness Floor)", fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_decision {
            label = "4. Automated Decision & Promotion Engine";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            DEC [label="Health Gatekeeper\\n(Error <= 1.0%, Faithfulness >= 0.90)", fillcolor="#581C87", color="#A855F7"];
            ACT_PROMO [label="Promote to 100% Production\\n(Update :prod pointer to Candidate)", fillcolor="#065F46", color="#059669"];
            ACT_ROLL [label="Instant Rollback (< 500ms)\\n(Revert Ingress Pointer to Stable)", fillcolor="#881337", color="#F43F5E"];
        }

        A1 -> GW [label="Resolves Prompt"];
        A2 -> GW [label="Routes Model Endpoints"];
        A3 -> GW [label="Instantiates Agent Graph"];

        GW -> SP_BLUE [label="Primary Route"];
        GW -> SP_GREEN [label="Canary Route"];

        SP_BLUE -> T1 [label="Emits Traces"];
        SP_GREEN -> T1 [label="Emits Canary Traces"];
        T1 -> T2 [label="Online Scoring"];

        T2 -> DEC [label="Streams Metrics"];
        DEC -> ACT_PROMO [label="Canary Passes"];
        DEC -> ACT_ROLL [label="Breached SLA"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Continuous Deployment
1. **Decouple Prompt Updates from Application Redeployments**: Hardcoding prompts in application source code requires a full container build, image push, and rolling cluster restart (taking 10–20 minutes) for a one-word prompt fix. Pull prompts dynamically from LangSmith Prompt Hub using versioned tags (`:prod`, `:staging`) to hot-swap prompts in milliseconds.
2. **Never Direct 100% Traffic to Unvalidated Endpoints**: Model provider changes can subtly alter JSON output formatting, token usage, or refusal behaviors. Always use progressive Canary traffic routing (e.g., 5% -> 25% -> 100%) before full promotion.
3. **Sub-Second Automated Rollbacks**: When telemetry alarms fire in production, rollback must not depend on human intervention or re-triggering CI/CD pipelines. AI Gateways must instantly switch routing pointers back to the prior known-good commit.
""")

st.divider()

# ==============================================================================
# SECTION 2: PROMPT DEPLOYMENT
# ==============================================================================
st.header("2. Prompt Deployment (Prompt Hub Tags: prod, staging; Dynamic Hot-Swapping without Redeploy)", divider="rainbow")

st.markdown("""
Prompt engineering in enterprise systems is an agile, high-frequency activity. Decoupling prompts from Python source code 
allows prompt engineers and product managers to iterate rapidly while maintaining rigorous GitOps version governance.
""")

st.markdown("""
### The Mechanics of LangSmith Prompt Hub Deployment
* **Immutable Commit Hashes vs Mutable Semantic Tags**: Every prompt push creates an immutable SHA commit (e.g., `hub.pull("org/support-prompt:8a9f3b1")`). For production applications, code references mutable semantic environment tags (`hub.pull("org/support-prompt:prod")` or `:staging`).
* **In-Memory TTL Caching**: Querying the Prompt Hub on every single incoming user HTTP request adds 50–150ms of network latency. Enterprise applications employ in-memory caching with a short Time-To-Live (e.g., 60 seconds) or dynamic webhook invalidation, delivering near-instant hot-swapping without runtime latency penalties.
* **Fallback Hardcoded Defaults**: Enterprise resilience dictates that if LangSmith Prompt Hub is temporarily unreachable due to network partition, the application falls back to a locally cached default template rather than failing user requests.
""")

st.code('''
from langsmith import Client
from langchain import hub
from functools import lru_cache

client = Client()

# Enterprise Dynamic Prompt Loader with Fallback
class DynamicPromptLoader:
    def __init__(self, prompt_handle: str = "acme-corp/enterprise-rag-prompt"):
        self.prompt_handle = prompt_handle
        self.active_tag = "prod"

    def fetch_production_prompt(self, environment: str = "prod"):
        """
        Fetches the active prompt template tagged with the target environment.
        Hot-swapping the tag in LangSmith instantly updates downstream behavior.
        """
        try:
            prompt_artifact = hub.pull(f"{self.prompt_handle}:{environment}")
            return prompt_artifact
        except Exception as e:
            # Fallback to local cached blueprint if network fails
            print(f"Warning: Prompt Hub pull failed ({e}). Reverting to local snapshot.")
            return self._local_fallback_prompt()

    def _local_fallback_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise AI assistant. Answer using provided context."),
            ("human", "{user_query}")
        ])
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern stacks, non-technical domain experts can test and refine prompts in the LangSmith Web UI sandbox. 
Once approved through automated Continuous Evaluation, a single click or automated GitHub Action shifts the `:prod` tag 
from commit `v1.8.2` to `v1.9.0`. All microservices reading `:prod` pick up the new prompt within 60 seconds without a single pod reboot.
""")

st.divider()

# ==============================================================================
# SECTION 3: MODEL DEPLOYMENT
# ==============================================================================
st.header("3. Model Deployment (Canary Traffic Splitting, Blue-Green Endpoint Swaps and Gateway Routing)", divider="rainbow")

st.markdown("""
Deploying a new model checkpoint—whether migrating from `gpt-4o` to `claude-3-5-sonnet`, or switching from FP16 to an 8-bit quantized 
vLLM endpoint—carries immense operational risk. Continuous Deployment orchestrates this via Canary traffic splitting and Blue-Green routing.
""")

st.markdown("""
### Traffic Routing Patterns for LLM Deployments
1. **Canary Traffic Splitting (e.g., 95% Blue / 5% Green)**: A percentage-based router forwards 5% of incoming user sessions to the candidate model (Green). LangSmith traces both streams into separate project views or tagged runs (`env:blue` vs `env:green`) to monitor latency, cost, and user satisfaction side-by-side.
2. **Blue-Green Instant Switchover**: Once the Green deployment proves stable and satisfies all SLAs over 24 hours, the gateway switches 100% of traffic to Green. Blue remains standing by for 1 hour to support immediate sub-second rollback if hidden edge cases surface.
3. **Shadow Routing (0% User Exposure)**: 100% of user traffic is handled by Blue, while the Gateway asynchronously duplicates requests to Green in the background. Green's answers are logged to LangSmith for offline evaluation and never shown to the user.
""")

routing_comparison = pd.DataFrame([
    {"Strategy": "Shadow Routing", "Live User Impact": "Zero (0%)", "Cost Multiplier": "2.0x (Runs Both)", "Failover Speed": "Instant", "Best Used For": "Major architecture rewrites & new LLM provider evaluations"},
    {"Strategy": "Canary Splitting (5-10%)", "Live User Impact": "Isolated (5%)", "Cost Multiplier": "1.05x", "Failover Speed": "< 1 second", "Best Used For": "Fine-tuned model checkpoint updates & quantized migrations"},
    {"Strategy": "Blue-Green Cutover", "Live User Impact": "100% on Switch", "Cost Multiplier": "1.0x (Sequential)", "Failover Speed": "< 500ms", "Best Used For": "Routine minor prompt and pipeline upgrades"},
    {"Strategy": "Direct In-Place Swap", "Live User Impact": "100% Immediate", "Cost Multiplier": "1.0x", "Failover Speed": "5-15 Minutes", "Best Used For": "Dev/Staging only; STRICTLY FORBIDDEN IN PRODUCTION"}
])
st.dataframe(routing_comparison, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production architectures place an AI Gateway (such as LiteLLM, Cloudflare AI Gateway, or Kong) in front of model endpoints. 
The gateway uses weighted round-robin or hash-based routing. The CD pipeline manages these weights programmatically: 
`deploy.canary(weight=5)` -> monitor telemetry -> `deploy.promote(weight=100)`.
""")

st.divider()

# ==============================================================================
# SECTION 4: AGENT DEPLOYMENT
# ==============================================================================
st.header("4. Agent Deployment (Compiled StateGraph Artifacts, Multi-Worker Container Scaling and State Continuity)", divider="rainbow")

st.markdown("""
Deploying autonomous agents built with LangGraph introduces stateful challenges not present in standard chains. 
Agents have multi-step state schemas, persistent thread checkpoints, and human-in-the-loop interrupts. Deploying an agent 
update without versioning the state schema will corrupt in-flight conversations and crash long-running tasks.
""")

st.markdown("""
### Core Principles of Safe Agent Deployment
* **Immutable State Schema Versioning**: Never mutate an existing TypedDict or Pydantic state schema in place. If an agent version requires new state fields, create `AgentStateV2`. Provide backward-compatible default values so active threads deserialized from the checkpointer (`SqliteSaver` or `PostgresSaver`) do not encounter `KeyError`.
* **Stateful Draining during Deployment**: Before destroying old worker pods during a rolling update, allow in-flight agent graphs to run to their next checkpoint or completion. New user threads are routed to the newly deployed container.
* **LangGraph Cloud / LangGraph Server Orchestration**: In production, deploy compiled graphs as independent web services exposed via the LangGraph Server API. This decouples the agent orchestration engine from client applications, managing thread queues and task persistence automatically.
""")

st.code('''
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator

# Versioned Agent State Schema with Default Value Resilience
class AgentStateV2(TypedDict):
    messages: List[str]
    intent: str
    user_authenticated: bool
    context_docs: List[str]
    schema_version: str  # Tracks state version for checkpointer migration

def build_production_agent():
    builder = StateGraph(AgentStateV2)
    # Define nodes and edges...
    # builder.add_node("planner", planner_node)
    # ...
    # Compile graph with persistent enterprise checkpointer
    compiled_agent = builder.compile()
    return compiled_agent

# Production Service Entrypoint
agent_app = build_production_agent()
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams containerize LangGraph applications using Docker and deploy them onto Kubernetes or LangGraph Cloud. 
Persistent checkpoints reside in managed PostgreSQL databases. When a new agent container is deployed, it automatically 
mounts the existing Postgres checkpointer, enabling seamless multi-turn conversations across pod lifecycles.
""")

st.divider()

# ==============================================================================
# SECTION 5: VERSION MANAGEMENT
# ==============================================================================
st.header("5. Version Management (SemVer Release Tracking, Environment Manifests and Lineage Auditing)", divider="rainbow")

st.markdown("""
When a production incident occurs at 2:00 AM, the first question on-call engineers ask is: 
*"What version of the prompt, model, embedding store, and code is currently running?"* 
Without rigorous version management, LLM applications become an untraceable matrix of ad-hoc configurations.
""")

st.markdown("""
### The 4 Pillars of GenAI Release Manifests
1. **Semantic Versioning (SemVer 2.0)**:
   * **Major (v2.0.0)**: Breaking change to input/output schema or complete model family migration (e.g., GPT to Claude).
   * **Minor (v1.1.0)**: Retrieval strategy changes, new tools added to agent, or major prompt revisions.
   * **Patch (v1.0.1)**: Few-shot example adjustments, prompt phrasing refinements, or hyperparameter tweaks.
2. **Environment Manifest File (`release_manifest.json`)**: Every deployment generates an immutable JSON manifest capturing exact commit SHAs across Git, LangSmith Prompt Hub, and VectorDB index timestamps.
3. **Trace Metadata Tagging**: LangSmith traces must automatically inject release metadata (`app_version`, `git_commit`, `prompt_commit`) into every root run to enable instant filtering by version.
""")

manifest_example = pd.DataFrame([
    {"Component": "Application Core", "Identifier": "customer-rag-service", "Release Version": "v2.4.1", "Commit / Hash": "git-sha-f82a91c"},
    {"Component": "System Prompt", "Identifier": "hub:support-persona", "Release Version": "v1.9.0", "Commit / Hash": "prompt-sha-3b7d1e0"},
    {"Component": "Primary LLM", "Identifier": "Azure OpenAI Service", "Release Version": "gpt-4o-2024-08-06", "Commit / Hash": "endpoint-prod-eastus"},
    {"Component": "Vector Index", "Identifier": "Pinecone / Qdrant", "Release Version": "kb-v4.2-snapshot", "Commit / Hash": "index-snap-992a01b"},
    {"Component": "LangGraph Schema", "Identifier": "AgentStateV2", "Release Version": "v2.0.0", "Commit / Hash": "schema-pydantic-b2e1"}
])
st.dataframe(manifest_example, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Release manifests are committed to Git as part of a GitOps workflow (ArgoCD or Flux). 
When a deployment pipeline executes, it tags the LangSmith project with the release version, creating an audit trail linking code commits directly to live trace telemetry.
""")

st.divider()

# ==============================================================================
# SECTION 6: ROLLBACKS
# ==============================================================================
st.header("6. Rollbacks (Instant Automated Rollbacks, One-Click Prompt Pinning and State Recovery)", divider="rainbow")

st.markdown("""
A deployment strategy is only as good as its rollback plan. In traditional software, rollbacks often require executing 
revert commits, building containers, and redeploying. For LLM applications in production, an active regression—such as 
a prompt update causing hallucinations or jailbreaks—requires sub-second mitigation.
""")

st.markdown("""
### Rollback Execution Modes
* **Automated Circuit Breaker Rollback (< 500ms)**: Real-time telemetry monitoring detects canary error rates exceeding 2.0% or latency exceeding 3000ms. The gateway automatically resets routing weights to 100% Blue without human intervention.
* **One-Click Prompt Pinning**: If a prompt change introduced subtle quality degradation that bypassed initial canary alarms, engineers can re-point the `:prod` tag in LangSmith Prompt Hub back to the previous known-good commit hash (`v1.8.0`) in a single click.
* **Stateful Agent Recovery**: If a newly deployed agent version causes state transition deadlocks, rollback directs traffic back to the prior container image, which resumes threads from the last valid checkpoint stored in the database.
""")

st.code('''
class AutomatedRollbackSupervisor:
    """
    Sub-second Circuit Breaker for LLM Continuous Deployment.
    """
    def __init__(self, ai_gateway_client, prompt_hub_client):
        self.gateway = ai_gateway_client
        self.hub = prompt_hub_client

    def evaluate_and_enforce_rollback(
        self,
        canary_error_rate: float,
        error_budget_ceiling: float = 0.02,
        stable_prod_version: str = "v1.8.0"
    ):
        if canary_error_rate > error_budget_ceiling:
            # 1. Instantly kill canary routing at the AI gateway
            self.gateway.update_routing_weights(canary_weight=0, stable_weight=100)
            
            # 2. Re-pin the prompt hub tag to known-good baseline
            self.hub.update_tag(tag="prod", target_commit=stable_prod_version)
            
            # 3. Broadcast critical incident notification
            self._notify_incident(
                severity="P0",
                message=f"Automated rollback triggered! Canary error rate {canary_error_rate*100:.1f}% exceeded budget."
            )
            return {"status": "ROLLBACK_EXECUTED", "active_version": stable_prod_version}
            
        return {"status": "HEALTHY"}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
High-performing engineering teams test automated rollbacks in staging using synthetic fault injection. 
Simulating a 5% error spike validates that the deployment manager automatically terminates the canary and restores stable traffic within 300 milliseconds.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: LLM Deployment Manager and Zero-Downtime Rollback Engine (LLMDeploymentManagerEngine)", divider="rainbow")

st.markdown("""
Below is the complete production-grade implementation of the deployment manager. 
It encapsulates canary rollout stages, dynamic tag management, real-time telemetry threshold verification, 
and automated zero-downtime rollback execution.
""")

st.code('''
class LLMDeploymentManagerEngine:
    """
    Enterprise Continuous Deployment & Zero-Downtime Rollback Engine.
    Emulates dynamic Prompt Hub tag switching, Canary traffic splitting,
    StateGraph compiled agent deployment, version lineage auditing, and automated rollback triggers.
    """
    def __init__(self, app_name: str = "enterprise-support-agent"):
        self.app_name = app_name
        self.deployment_history: List[Dict[str, Any]] = []

    def deploy_release(
        self,
        target_version: str,
        prompt_tag: str,
        model_endpoint: str,
        traffic_split_percent: int,
        canary_error_threshold_percent: float = 2.0,
        inject_canary_outage: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a canary deployment stage, simulating health telemetry and automated rollback criteria.
        """
        # Baseline production state
        active_prod_version = "v1.8.0-stable"
        active_prod_prompt_tag = "prod"
        active_prod_endpoint = "gpt-4o-2024-08-06"

        # Telemetry generation
        if inject_canary_outage:
            canary_error_rate = 5.8
            canary_p95_ms = 4120.0
            rollback_triggered = True
            rollback_reason = f"Canary error rate {canary_error_rate:.1f}% breached threshold {canary_error_threshold_percent:.1f}%."
            final_status = "AUTOMATIC_ROLLBACK_EXECUTED"
            active_live_version = active_prod_version
            active_live_prompt_tag = active_prod_prompt_tag
            active_live_endpoint = active_prod_endpoint
            effective_traffic_split = 0
        else:
            canary_error_rate = 0.4
            canary_p95_ms = 1640.0
            rollback_triggered = False
            rollback_reason = "None"
            final_status = "CANARY_HEALTHY" if traffic_split_percent < 100 else "FULLY_PROMOTED_TO_PRODUCTION"
            active_live_version = target_version
            active_live_prompt_tag = prompt_tag
            active_live_endpoint = model_endpoint
            effective_traffic_split = traffic_split_percent

        report = {
            "deployment_id": f"deploy-{target_version.replace('.', '_')}",
            "target_version": target_version,
            "prompt_tag": prompt_tag,
            "model_endpoint": model_endpoint,
            "traffic_split_percent": effective_traffic_split,
            "canary_telemetry": {
                "error_rate_percent": canary_error_rate,
                "p95_latency_ms": canary_p95_ms,
                "sample_requests_evaluated": 1500
            },
            "rollback_triggered": rollback_triggered,
            "rollback_reason": rollback_reason,
            "final_status": final_status,
            "active_live_version": active_live_version,
            "active_live_prompt_tag": active_live_prompt_tag,
            "active_live_endpoint": active_live_endpoint
        }
        self.deployment_history.append(report)
        return report
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Blue-Green Canary Deployment Simulator and Instant Rollback Sandbox", divider="rainbow")

st.markdown("""
Simulate enterprise canary rollouts, test traffic splitting across candidate models, and verify 
how automated circuit breakers detect errors and instantly execute zero-downtime rollbacks.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Deployment Orchestration Controls")
    cand_version_select = st.selectbox(
        "Candidate Release Target",
        ["v1.9.0-rc1 (Upgraded Reasoning Prompt)", "v2.0.0-rc2 (Migrating to Claude 3.5 Sonnet)", "v1.9.1-hotfix (Regex Guard Patch)"],
        index=0
    )
    prompt_tag_input = st.selectbox(
        "Prompt Hub Tag",
        [":canary", ":staging", ":prod-candidate"],
        index=0
    )
    model_endpoint_input = st.selectbox(
        "Model Routing Endpoint",
        ["azure-openai-eastus/gpt-4o-2024-11", "anthropic-bedrock/claude-3-5-sonnet-v2", "vllm-internal-cluster/llama-3-3-70b"],
        index=0
    )
    traffic_split_slider = st.slider(
        "Canary Traffic Split Percentage (%)",
        min_value=0,
        max_value=100,
        value=15,
        step=5,
        help="Percentage of production user traffic routed to the candidate release."
    )
    err_threshold_slider = st.slider(
        "Max Allowed Canary Error Rate (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    simulate_outage_toggle = st.toggle(
        "Simulate Canary Spike / Outage (Fault Injection)",
        value=False,
        help="Simulates model provider degradation or runtime syntax crash triggering the circuit breaker."
    )

with wb_col2:
    st.markdown("### Live Deployment Status & Telemetry Monitor")
    clean_ver = cand_version_select.split(" ")[0]
    deploy_result = st.session_state.deploy_manager_engine.deploy_release(
        target_version=clean_ver,
        prompt_tag=prompt_tag_input,
        model_endpoint=model_endpoint_input,
        traffic_split_percent=traffic_split_slider,
        canary_error_threshold_percent=err_threshold_slider,
        inject_canary_outage=simulate_outage_toggle
    )

    if deploy_result["rollback_triggered"]:
        st.error(f"🚨 **DEPLOYMENT STATUS: {deploy_result['final_status']}**")
        st.markdown(f"**Circuit Breaker Reason:** {deploy_result['rollback_reason']}")
        st.warning(f"**Safety Action:** Ingress traffic immediately rerouted to baseline `{deploy_result['active_live_version']}` with Prompt Tag `{deploy_result['active_live_prompt_tag']}`.")
    else:
        st.success(f"✅ **DEPLOYMENT STATUS: {deploy_result['final_status']}**")
        st.markdown(f"Active Live Version: `{deploy_result['active_live_version']}` | Traffic Split: **{deploy_result['traffic_split_percent']}% Canary / {100 - deploy_result['traffic_split_percent']}% Stable**")

    # Telemetry Metrics Grid
    tel = deploy_result["canary_telemetry"]
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric(
            "Canary Error Rate",
            f"{tel['error_rate_percent']:.1f}%",
            delta=f"{tel['error_rate_percent'] - 0.4:+.1f}%",
            delta_color="inverse"
        )
    with tc2:
        st.metric(
            "p95 Latency",
            f"{tel['p95_latency_ms']:.0f} ms",
            delta=f"{tel['p95_latency_ms'] - 1640.0:+.0f} ms",
            delta_color="inverse"
        )
    with tc3:
        st.metric(
            "Canary Samples Evaluated",
            f"{tel['sample_requests_evaluated']:,}",
            delta="Online Stream"
        )

    # Ingress Route Table
    st.markdown("#### Live Ingress Route Table")
    route_df = pd.DataFrame([
        {"Route Name": "Stable Production (Blue)", "Target Version": "v1.8.0-stable", "Endpoint": "gpt-4o-2024-08-06", "Traffic Weight": f"{100 - deploy_result['traffic_split_percent']}%", "Health": "HEALTHY"},
        {"Route Name": "Canary Candidate (Green)", "Target Version": clean_ver, "Endpoint": model_endpoint_input, "Traffic Weight": f"{deploy_result['traffic_split_percent']}%", "Health": "DEGRADED" if deploy_result['rollback_triggered'] else "HEALTHY"}
    ])
    st.dataframe(route_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

q_exp = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with q_exp:
    st.markdown("""
    **Q1: Why is separating prompt deployment from code deployment critical in production GenAI architectures?**  
    *Answer:* Code redeployments require full CI/CD image builds, container registry pushes, and rolling pod restarts that take 10–20 minutes and introduce cluster disruption. Prompts require frequent calibration and hot-fixing. Decoupling prompts via the LangSmith Prompt Hub using dynamic tags (`:prod`, `:staging`) allows prompt engineers to hot-swap system instructions in under a second without restarting microservices.
    """)

    st.markdown("""
    **Q2: How does Canary Traffic Splitting mitigate risk during large language model migrations?**  
    *Answer:* Migration between model families (e.g., from GPT-4o to Claude 3.5 Sonnet) introduces behavioral discrepancies in tool-calling format, formatting compliance, and latency. Routing 5% of user traffic to the candidate allows real-time telemetry observation in LangSmith. If unexpected tool parsing errors or refusal spikes occur, the blast radius is confined to 5% of users and automated rollbacks instantly restore the baseline.
    """)

    st.markdown("""
    **Q3: What unique challenges arise when deploying stateful LangGraph agents compared to stateless chains?**  
    *Answer:* LangGraph agents persist execution checkpoints in databases across multi-turn sessions. If a new deployment mutates the state schema without backward-compatible defaults, active in-flight threads deserialized from the checkpointer will crash. Zero-downtime agent deployment requires immutable state schema versioning (`AgentStateV2`) and graceful draining of in-flight threads.
    """)

    st.markdown("""
    **Q4: What are the three essential components of an automated rollback supervisor?**  
    *Answer:* 1) Real-time telemetry monitoring (error rate, p95 latency, guardrail violation rate); 2) A dynamic routing controller (AI Gateway API to reset canary weights to 0%); and 3) An immutable fallback pointer (a pinned Prompt Hub commit hash and model checkpoint tag verified to be stable).
    """)

    st.markdown("""
    **Q5: What is an Environment Release Manifest, and why is it mandatory for audit compliance?**  
    *Answer:* A release manifest is an immutable document recording the exact versions of all composite artifacts powering an LLM service at release time: application Git commit SHA, Prompt Hub commit hash, model endpoint identifier, vector index snapshot ID, and evaluation benchmark run ID. It provides full lineage traceability for compliance, security forensics, and incident post-mortems.
    """)

st.markdown("### Production Traps and Anti-Patterns")

tr_col1, tr_col2 = st.columns(2)

with tr_col1:
    st.error("""
    **Trap 1: Mutating Active Prompt Tags Directly in Production**
    * **Failure**: Manually editing the live `:prod` prompt template in the web UI without testing in `:staging`.
    * **Consequence**: A syntax typo or missing template variable immediately breaks 100% of live customer requests.
    * **Remedy**: Always push prompt changes to `:staging`, run automated Continuous Evaluation, and promote to `:prod` via verified pipelines.
    """)

    st.error("""
    **Trap 2: Big-Bang 100% Cutover on Model Checkpoint Swaps**
    * **Failure**: Instantly redirecting 100% of production traffic to a newly released model endpoint without canary staging.
    * **Consequence**: The new model experiences unexpected regional rate limits or subtle JSON schema formatting differences, causing global service outage.
    * **Remedy**: Enforce progressive canary traffic rollouts (5% -> 25% -> 100%) with automated error rate circuit breakers.
    """)

with tr_col2:
    st.warning("""
    **Trap 3: In-Memory Prompt Stale Cache Deadlocks**
    * **Failure**: Implementing local prompt caching with an infinite TTL and no cache invalidation mechanism.
    * **Consequence**: Applications continue serving outdated prompt templates hours after an emergency rollback was triggered in LangSmith.
    * **Remedy**: Use short TTLs (e.g., 60 seconds) or configure webhook-driven cache invalidation endpoints on application servers.
    """)

    st.warning("""
    **Trap 4: Breaking State Schemas in LangGraph Persistent Checkpoints**
    * **Failure**: Renaming or deleting state keys in a compiled StateGraph without backward-compatible fallback values.
    * **Consequence**: Existing customer multi-turn conversations stored in PostgreSQL crash with `KeyError` upon resuming their thread.
    * **Remedy**: Make all state schema additions optional or provide explicit default values for deprecated keys during migrations.
    """)

st.markdown("### Core Technical Flashcards")

fc_col1, fc_col2 = st.columns(2)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: How does LangSmith Prompt Hub support zero-downtime prompt updates?")
        st.markdown("**Back**: By pulling prompts dynamically via mutable semantic tags (`:prod`, `:staging`); changing the pointer hot-swaps behavior without redeploying containers.")

    with st.container(border=True):
        st.markdown("**Front**: What is the primary advantage of Shadow Routing over Canary Deployment?")
        st.markdown("**Back**: Shadow Routing has zero live user impact (candidate outputs are discarded), whereas Canary Deployment exposes 5-10% of real users to the candidate.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is an automated circuit breaker in LLM deployment pipelines?")
        st.markdown("**Back**: An automated supervisor that monitors real-time canary error rates and instantly resets gateway traffic weights to 0% if error budgets are breached.")

    with st.container(border=True):
        st.markdown("**Front**: Why must StateGraph state schemas be backward-compatible during updates?")
        st.markdown("**Back**: Because persistent checkpointers reload active multi-turn session states serialized under older schemas; missing fields cause runtime deserialization crashes.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Continuous Deployment for LLM applications:")

cd_chk1 = st.checkbox("I understand how to decouple prompts from application binaries using LangSmith Prompt Hub tags.", value=True)
cd_chk2 = st.checkbox("I know how to design Canary Traffic Splitting architectures using AI Gateways.", value=True)
cd_chk3 = st.checkbox("I understand the backward-compatibility rules required to safely deploy stateful LangGraph agents.", value=True)
cd_chk4 = st.checkbox("I can generate immutable Release Manifests linking Git SHAs, Prompt hashes, and Model endpoints.", value=True)
cd_chk5 = st.checkbox("I know how to implement automated circuit breakers for sub-second rollback execution.", value=True)
cd_chk6 = st.checkbox("I understand the tradeoffs between Shadow Routing, Canary Splitting, and Blue-Green Cutover.", value=True)

if cd_chk1 and cd_chk2 and cd_chk3 and cd_chk4 and cd_chk5 and cd_chk6:
    st.success("Mastery Certified: You have mastered Continuous Deployment, Canary traffic routing, and Zero-Downtime Rollbacks for enterprise LLM systems.")
