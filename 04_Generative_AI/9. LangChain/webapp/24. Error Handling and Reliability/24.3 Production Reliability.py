import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.3 Production Reliability - LangChain Master Curriculum",
    page_icon="🏗️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ProductionReliabilityGovernorEngine:
    """
    Enterprise Production Reliability Governor Engine for LangChain & LangGraph systems.
    Evaluates and simulates the 7 mission-critical architectural pillars:
    1. Idempotency (Deduplication keys, deterministic replay of stateful side-effects)
    2. State Recovery (Hydrating conversation graph state from cold persistence)
    3. Durable Execution (Workflows surviving worker pod crashes and restarts)
    4. Checkpointing (Differential state snapshots, MemorySaver vs PostgresSaver)
    5. Failure Recovery (Dead-letter queues, automated failover, and compensation flows)
    6. Circuit Breakers (CLOSED, OPEN, HALF-OPEN state machines protecting external APIs)
    7. Rate Limiting (Token-bucket rate governors and tenant concurrency semaphores)
    """
    def __init__(self, service_name: str = "enterprise-rag-governor"):
        self.service_name = service_name

    def simulate_production_governance(
        self,
        request_burst_count: int = 30,
        provider_failure_spike_pct: float = 40.0,
        breaker_error_threshold_pct: float = 25.0,
        enable_idempotency: bool = True,
        enable_checkpointing: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Simulates enterprise traffic handling with circuit breakers and state checkpointing.
        """
        rng = np.random.default_rng(random_seed)

        # Circuit Breaker states: CLOSED (normal), OPEN (tripped/failing fast), HALF_OPEN (probing)
        breaker_state = "CLOSED"
        consecutive_failures = 0
        total_executed = 0
        total_short_circuited = 0
        total_deduplicated = 0
        checkpoints_persisted = 0
        successful_executions = 0
        failed_executions = 0

        trace_log = []
        seen_idempotency_keys = set()

        for req_idx in range(1, request_burst_count + 1):
            # Synthesize idempotency key (introduce 20% duplicate client retries)
            is_duplicate = (req_idx % 5 == 0) and (req_idx > 5)
            idempotency_key = f"key-dedup-{(req_idx - 1) if is_duplicate else req_idx:03d}"

            # Step 1: Idempotency Gate
            if enable_idempotency and (idempotency_key in seen_idempotency_keys):
                total_deduplicated += 1
                trace_log.append({
                    "Req #": f"#{req_idx}",
                    "Idempotency Key": idempotency_key,
                    "Circuit Breaker": breaker_state,
                    "Action Taken": "Cached Replay (Deduplicated)",
                    "Status": "CACHED_OK",
                    "Latency (ms)": 8.0,
                    "Details": "Identical client request key detected. Returning previously committed result without invoking LLM."
                })
                continue

            seen_idempotency_keys.add(idempotency_key)

            # Step 2: Circuit Breaker Inspection
            if breaker_state == "OPEN":
                # Check if cooling interval elapsed (simulate probe every 5 requests)
                if req_idx % 4 == 0:
                    breaker_state = "HALF_OPEN"
                else:
                    total_short_circuited += 1
                    trace_log.append({
                        "Req #": f"#{req_idx}",
                        "Idempotency Key": idempotency_key,
                        "Circuit Breaker": breaker_state,
                        "Action Taken": "Fast Fail (Short-Circuited)",
                        "Status": "CIRCUIT_BLOCKED",
                        "Latency (ms)": 2.0,
                        "Details": "Circuit breaker is OPEN. Fast failing to secondary provider or returning cached fallback immediately."
                    })
                    continue

            # Step 3: Execution under Chaos Conditions
            total_executed += 1
            is_failure = (rng.uniform(0, 100) < provider_failure_spike_pct)

            if is_failure:
                failed_executions += 1
                consecutive_failures += 1
                observed_err_rate = (failed_executions / total_executed) * 100.0

                if observed_err_rate >= breaker_error_threshold_pct and total_executed >= 5:
                    breaker_state = "OPEN"

                trace_log.append({
                    "Req #": f"#{req_idx}",
                    "Idempotency Key": idempotency_key,
                    "Circuit Breaker": breaker_state,
                    "Action Taken": "LLM Invocations Failed (HTTP 503)",
                    "Status": "FAILED",
                    "Latency (ms)": 450.0,
                    "Details": f"Provider dropped connection. Consecutive fails: {consecutive_failures}. Breaker state: {breaker_state}."
                })
            else:
                successful_executions += 1
                consecutive_failures = 0
                if breaker_state == "HALF_OPEN":
                    breaker_state = "CLOSED"  # Probing call succeeded, reset breaker

                # Step 4: Checkpointing state
                if enable_checkpointing:
                    checkpoints_persisted += 1

                trace_log.append({
                    "Req #": f"#{req_idx}",
                    "Idempotency Key": idempotency_key,
                    "Circuit Breaker": breaker_state,
                    "Action Taken": "Model Success & State Checkpointed",
                    "Status": "SUCCESS_200",
                    "Latency (ms)": 820.0,
                    "Details": f"Executed cleanly. Thread snapshot written to persistent PostgresSaver checkpoint."
                })

        return {
            "request_count": request_burst_count,
            "breaker_final_state": breaker_state,
            "total_executed": total_executed,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "total_short_circuited": total_short_circuited,
            "total_deduplicated": total_deduplicated,
            "checkpoints_persisted": checkpoints_persisted,
            "trace_log": trace_log
        }


# ==============================================================================
# HEADER BLOCK
# ==============================================================================
st.title("24.3 Production Reliability: Enterprise Durability, Checkpointing, and Governance")
st.caption("LangChain v0.3 & LangGraph Enterprise Operations • Idempotency Keys, State Checkpointing, Circuit Breakers, Durable Execution, and Rate Throttling")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Production Reliability and Mission-Critical Governor Architecture", divider="rainbow")

st.markdown("""
In enterprise deployments, LLM applications transition from single-turn request-reply scripts to long-running, 
stateful autonomous workflows. In these environments, failures are inevitable: Kubernetes worker pods are evicted, 
upstream LLM providers experience severe latency spikes, and end users retry queries repeatedly.

An enterprise production reliability governor implements five protective boundaries:
1. **Idempotency Deduplication**: Ensuring financial transactions or tool mutations execute exactly once, even during aggressive network retries.
2. **Persistent Checkpointing**: Serializing execution state (threads, message lists, tool scratchpads) to durable databases after every node transition.
3. **Durable Asynchronous Queues**: Allowing workflows to survive server crashes and resume from the exact point of interruption.
4. **Adaptive Circuit Breakers**: Automatically isolating degraded APIs to protect systems from cascading failure.
5. **Token-Bucket Rate Governors**: Throttling client requests to prevent unexpected quota exhaustion.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ProductionReliabilityBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=11, fontcolor="#F8FAFC", style="filled", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_ingress_governance {
            label = "Tier 1: Ingress Governance & Concurrency Control";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            client_traffic [label="Client HTTP Request\\n(With Idempotency-Key Header)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            idempotency_cache [label="Idempotency Cache & Lock\\n(Redis SETNX with TTL)", shape=diamond, fillcolor="#B45309", color="#F59E0B"];
            token_bucket [label="Token-Bucket Rate Limiter\\n(Tenant RPM/TPM Throttling)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_orchestration_engine {
            label = "Tier 2: Stateful Workflow Orchestrator (LangGraph / Temporal)";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            workflow_runtime [label="LangGraph StateGraph Runtime\\n(Cyclic Execution Engine)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            node_executor [label="Graph Node Invocation\\n(LLM Call / Tool Dispatch)", shape=box, fillcolor="#4C1D95", color="#8B5CF6"];
            checkpoint_saver [label="Durable Checkpointer\\n(PostgresSaver / SqliteSaver)", shape=cylinder, fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_resilience_circuit {
            label = "Tier 3: Circuit Breaker & Provider Gateway";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            circuit_breaker [label="Adaptive Circuit Breaker\\n(Closed -> Open -> Half-Open)", shape=octagon, fillcolor="#991B1B", color="#EF4444"];
            provider_api [label="External LLM / Tool Provider\\n(OpenAI / Third-Party REST)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            dlq_store [label="Dead-Letter Queue (DLQ)\\n(Post-Mortem & Alert Buffer)", shape=cylinder, fillcolor="#991B1B", color="#EF4444"];
        }

        subgraph cluster_egress_tier {
            label = "Tier 4: Durable Delivery & State Time-Travel";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            final_response [label="Validated Durable Response\\n(Commit State & Release Lock)", shape=box, fillcolor="#047857", color="#10B981"];
            human_interrupt [label="Human-in-the-Loop Interrupt\\n(Awaiting Review / Approval)", shape=box, fillcolor="#B45309", color="#F59E0B"];
        }

        client_traffic -> idempotency_cache;
        idempotency_cache -> final_response [label="Duplicate Key (Replay Cached)"];
        idempotency_cache -> token_bucket [label="New Key (Acquired)"];
        token_bucket -> workflow_runtime [label="Tokens Granted"];

        workflow_runtime -> node_executor [label="Next Step"];
        node_executor -> circuit_breaker [label="Outbound Invocation"];
        circuit_breaker -> provider_api [label="State=CLOSED"];
        circuit_breaker -> dlq_store [label="State=OPEN (Tripwire)"];

        provider_api -> node_executor [label="Response 200"];
        node_executor -> checkpoint_saver [label="Persist State Snapshot"];
        checkpoint_saver -> workflow_runtime [label="Thread Hydrated"];

        workflow_runtime -> human_interrupt [label="Sensitive Action (interrupt())"];
        human_interrupt -> workflow_runtime [label="Resume with Action"];
        workflow_runtime -> final_response [label="Workflow Terminal Node"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: IDEMPOTENCY
# ==============================================================================
st.header("2. Idempotency (Deduplication Keys, Deterministic Cache Replay and Safe Financial Actions)", divider="rainbow")

st.markdown("""
In distributed systems, an operation is **idempotent** if performing it multiple times yields the exact same outcome 
as performing it once. In LLM applications, idempotency is crucial because models are non-deterministic, and agents 
are frequently equipped with side-effect tools (such as charging credit cards, writing to databases, or sending emails).

If an HTTP network timeout causes a client to retry a request, a naive system will execute the tool twice, 
charging the user double.

**The Idempotency Key Pattern**:
1. The client sends a unique header: `Idempotency-Key: uuid-4-xxxx`.
2. The application checks a distributed lock (e.g., Redis `SETNX`).
3. If the key exists and execution is pending, concurrent duplicate requests wait or fail fast.
4. If execution completed, the cached response is replayed immediately without invoking the LLM or tools again.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production LangServe and FastAPI gateways enforce idempotency via Redis middleware. 
Idempotency keys have a standard Time-To-Live (TTL) of 24 hours, guaranteeing that retry bursts do not trigger duplicate mutations.
""")

st.divider()

# ==============================================================================
# SECTION 3: STATE RECOVERY
# ==============================================================================
st.header("3. State Recovery (Persistent Checkpoint Ingestion, Thread Deserialization and Resumed Execution)", divider="rainbow")

st.markdown("""
Traditional LangChain chains store intermediate state in Python memory. If the server process dies, crashes due to an Out-Of-Memory (OOM) 
condition, or is terminated during a rolling Kubernetes deployment, all active conversation histories and reasoning scratchpads vanish.

**State Recovery Architecture**:
- Conversation threads are partitioned by persistent **Thread IDs** (`thread_id="tenant_123_conv_456"`).
- State transitions are serialized into type-safe JSON or binary protocol buffers.
- When a worker boots up, it queries the persistent storage layer by `thread_id`, deselects completed steps, 
  and reconstructs the exact graph execution state without requiring the user to start over.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern agent systems decouple runtime execution from process memory. Every agent step updates an append-only event log. 
If an execution pod crashes midway through a 5-step agent loop, a replacement pod hydrates the state from the log 
and resumes execution at step 3.
""")

st.divider()

# ==============================================================================
# SECTION 4: DURABLE EXECUTION
# ==============================================================================
st.header("4. Durable Execution (Workflows that Survive Server Restarts, Task Queues and Asynchronous Workers)", divider="rainbow")

st.markdown("""
**Durable Execution** guarantees that code execution proceeds to completion regardless of hardware failures, 
network timeouts, or deployment restarts. This capability is essential for long-running workflows that span 
minutes, hours, or days (such as multi-document research analysis or human-approval pipelines).

Key building blocks of durable execution:
- **Asynchronous Task Queues**: Tasks are submitted to durable brokers (Celery, Temporal, or AWS SQS) rather than running in HTTP request threads.
- **Deterministic Replay**: The workflow runtime replays past node events from storage rather than re-executing completed operations.
- **Non-Blocking Sleep**: Pausing workflows using timers without holding open active threads or consuming server memory.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise AI teams deploy LangGraph combined with durable orchestration engines like Temporal or AWS Step Functions, 
ensuring that complex 20-step multi-agent research pipelines finish successfully even across rolling cluster upgrades.
""")

st.divider()

# ==============================================================================
# SECTION 5: CHECKPOINTING
# ==============================================================================
st.header("5. Checkpointing (LangGraph Checkpointers: MemorySaver vs PostgresSaver, Thread Time Travel)", divider="rainbow")

st.markdown("""
In LangGraph v0.2+, **Checkpointers** are first-class architectural components that capture point-in-time snapshots 
of the application state after every graph node execution:

| Feature | `MemorySaver` | `PostgresSaver` | `SqliteSaver` |
| :--- | :--- | :--- | :--- |
| **Storage Medium** | In-Memory Python Dict | PostgreSQL Relational DB | Local Disk SQLite File |
| **Persistence Across Reboots** | ❌ Lost on process exit | ✅ Fully durable | ✅ Durable locally |
| **Horizontal Scaling** | ❌ Bound to single pod | ✅ Multi-worker cluster ready | ❌ Single-instance only |
| **Production Fit** | Prototyping & Unit Tests | Enterprise Production | Edge & Desktop Apps |

**Time Travel**:
Because checkpointers retain historical state versions, engineers can rewind a conversation thread to an earlier state, 
modify inputs, and replay from that point to debug errors or inspect alternative reasoning paths.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production LangGraph deployments connect `PostgresSaver` with connection pools (`psycopg_pool`), saving diff snapshots 
at each step. This enables seamless Human-in-the-Loop workflows where execution pauses for hours waiting for a human manager's review.
""")

st.divider()

# ==============================================================================
# SECTION 6: FAILURE RECOVERY
# ==============================================================================
st.header("6. Failure Recovery (Automated Failover Routines, Dead-Letter Queues and Graceful Degradation)", divider="rainbow")

st.markdown("""
When all retries and fallback models fail, enterprise systems rely on structured failure recovery:

1. **Dead-Letter Queues (DLQ)**: Failed request payloads, complete trace IDs, and stack traces are routed to a DLQ topic (Kafka / SQS) for automated triage or manual engineer replay.
2. **Compensation Transactions**: If a workflow fails midway (e.g., flight booked, but hotel booking failed), compensating reverse actions (cancel flight) are executed automatically to prevent corrupted state.
3. **Graceful UI Degradation**: The client receives a clean, actionable response: 
   `"We could not fetch live inventory right now, but your request has been queued and you will be notified shortly."`
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Failures are never silently dropped. When a dead-letter event occurs, an alert triggers on Slack, 
and the failed request is added to an automated evaluation suite to reproduce and test the fix.
""")

st.divider()

# ==============================================================================
# SECTION 7: CIRCUIT BREAKERS
# ==============================================================================
st.header("7. Circuit Breakers (Automated Tripwires on Error Bursts, Half-Open Probing and Gateway Failover)", divider="rainbow")

st.markdown("""
The **Circuit Breaker** pattern prevents an application from repeatedly calling a failing downstream dependency:

- **CLOSED State**: Normal operation. All requests pass through to the provider. Error rates are monitored over a sliding window.
- **OPEN State**: When the error rate exceeds the threshold (e.g., 25% failures over 20 requests), the circuit trips OPEN. 
  Subsequent requests fail fast immediately without hitting the provider, preventing resource exhaustion and high cloud bills.
- **HALF-OPEN State**: After a cooldown duration (e.g., 30 seconds), the circuit lets a limited probe batch (e.g., 5 requests) through. 
  If the probe succeeds, the circuit resets to CLOSED. If it fails, the circuit returns to OPEN.
""")

with st.expander("🔬 Circuit Breaker State Transition Diagram"):
    cb_dot = """
    digraph CircuitBreaker {
        rankdir=LR;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=10, fontcolor="#F8FAFC", style="filled", penwidth=1.2];
        edge [fontname="Arial", fontsize=9, fontcolor="#94A3B8", color="#38BDF8"];

        closed [label="CLOSED\\n(Normal Traffic Flow)", shape=ellipse, fillcolor="#047857", color="#10B981"];
        opened [label="OPEN\\n(Fail Fast / Reroute)", shape=ellipse, fillcolor="#991B1B", color="#EF4444"];
        half_open [label="HALF-OPEN\\n(Canary Probing)", shape=ellipse, fillcolor="#B45309", color="#F59E0B"];

        closed -> opened [label="Error Rate > 25%"];
        opened -> half_open [label="Cooldown Expired"];
        half_open -> closed [label="Canary Succeeded"];
        half_open -> opened [label="Canary Failed"];
    }
    """
    st.graphviz_chart(cb_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 8: RATE LIMITING
# ==============================================================================
st.header("8. Rate Limiting (Token Bucket Governors, Tenant Tier Throttling and Concurrency Semaphores)", divider="rainbow")

st.markdown("""
Production LLM APIs require two layers of rate limiting:
1. **Outbound Rate Limiting (Provider Quotas)**: Preventing the application from exceeding vendor TPM/RPM limits (e.g., 10,000 RPM on OpenAI Tier 4).
2. **Ingress Rate Limiting (Fair Share Multi-Tenancy)**: Preventing a single heavy tenant from exhausting the shared pool of tokens and degrading service for others.

**Algorithms**:
- **Token Bucket**: Allows bursts of traffic while enforcing a smooth steady-state processing rate.
- **Leaky Bucket**: Enforces a constant, uniform processing output rate.
- **Concurrency Semaphores**: Bounding simultaneous live connections to GPU instances or vector indices.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Rate limiting is typically enforced at the API gateway level using Redis and Lua scripts, 
returning standard `HTTP 429 Too Many Requests` responses with a `Retry-After: 30` header when limits are reached.
""")

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Production Reliability Governor Engine (ProductionReliabilityGovernorEngine)", divider="rainbow")

st.markdown("""
The following complete Python implementation demonstrates `ProductionReliabilityGovernorEngine`, an enterprise-grade 
reliability controller that coordinates idempotency deduplication, state checkpointing, and adaptive circuit breakers.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any

class ProductionReliabilityGovernorEngine:
    """
    Enterprise Production Reliability Governor Engine for LangChain & LangGraph systems.
    """
    def __init__(self, service_name: str = "enterprise-rag-governor"):
        self.service_name = service_name

    def simulate_production_governance(
        self,
        request_burst_count: int = 30,
        provider_failure_spike_pct: float = 40.0,
        breaker_error_threshold_pct: float = 25.0,
        enable_idempotency: bool = True,
        enable_checkpointing: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(random_seed)
        breaker_state = "CLOSED"
        total_executed = 0
        total_short_circuited = 0
        total_deduplicated = 0
        checkpoints_persisted = 0
        successful_executions = 0
        failed_executions = 0

        trace_log = []
        seen_keys = set()

        for req_idx in range(1, request_burst_count + 1):
            is_duplicate = (req_idx % 5 == 0) and (req_idx > 5)
            key = f"key-dedup-{(req_idx - 1) if is_duplicate else req_idx:03d}"

            if enable_idempotency and (key in seen_keys):
                total_deduplicated += 1
                trace_log.append({"Req #": f"#{req_idx}", "Action": "Cached Replay (Deduplicated)", "Status": "CACHED_OK"})
                continue

            seen_keys.add(key)

            if breaker_state == "OPEN":
                if req_idx % 4 == 0:
                    breaker_state = "HALF_OPEN"
                else:
                    total_short_circuited += 1
                    trace_log.append({"Req #": f"#{req_idx}", "Action": "Fast Fail (Short-Circuited)", "Status": "CIRCUIT_BLOCKED"})
                    continue

            total_executed += 1
            is_failure = (rng.uniform(0, 100) < provider_failure_spike_pct)

            if is_failure:
                failed_executions += 1
                if (failed_executions / total_executed) * 100.0 >= breaker_error_threshold_pct:
                    breaker_state = "OPEN"
                trace_log.append({"Req #": f"#{req_idx}", "Action": "Model Invocations Failed", "Status": "FAILED"})
            else:
                successful_executions += 1
                if breaker_state == "HALF_OPEN":
                    breaker_state = "CLOSED"
                if enable_checkpointing:
                    checkpoints_persisted += 1
                trace_log.append({"Req #": f"#{req_idx}", "Action": "Success & State Checkpointed", "Status": "SUCCESS_200"})

        return {
            "breaker_final_state": breaker_state,
            "total_executed": total_executed,
            "successful_executions": successful_executions,
            "total_short_circuited": total_short_circuited,
            "total_deduplicated": total_deduplicated,
            "checkpoints_persisted": checkpoints_persisted,
            "trace_log": trace_log
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Production Resilience Cockpit and Circuit Breaker Simulator", divider="rainbow")

st.markdown("""
Test how your system responds to traffic bursts, provider outages, and network retries. 
Observe the circuit breaker trip open, watch duplicate requests get absorbed by idempotency keys, 
and monitor persistent checkpoints being saved in real time.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    burst_count = st.slider("Request Burst Volume", min_value=10, max_value=60, value=30, step=5)
    failure_rate = st.slider("Downstream Provider Outage Rate (%)", min_value=0.0, max_value=100.0, value=45.0, step=5.0)
    breaker_threshold = st.selectbox("Circuit Breaker Error Tripwire Threshold", options=[15.0, 25.0, 40.0], index=1)

with col_wb2:
    enable_dedup = st.toggle("Enable Idempotency Key Deduplication (Redis Lock)", value=True)
    enable_ckpt = st.toggle("Enable Durable State Checkpointing (PostgresSaver)", value=True)

# Run simulation
gov_engine = ProductionReliabilityGovernorEngine()
gov_results = gov_engine.simulate_production_governance(
    request_burst_count=burst_count,
    provider_failure_spike_pct=failure_rate,
    breaker_error_threshold_pct=breaker_threshold,
    enable_idempotency=enable_dedup,
    enable_checkpointing=enable_ckpt
)

st.subheader("🛡️ Production Reliability Signals")

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
m_c1.metric("Final Breaker State", gov_results["breaker_final_state"])
m_c2.metric("Successful Runs", f"{gov_results['successful_executions']}")
m_c3.metric("Deduplicated Retries", f"{gov_results['total_deduplicated']}")
m_c4.metric("Short-Circuited Calls", f"{gov_results['total_short_circuited']}")

if gov_results["breaker_final_state"] == "OPEN":
    st.error("🚨 CIRCUIT BREAKER TRIPPED OPEN: Downstream error rate breached thresholds. All non-critical traffic is short-circuited to protect downstream services.")
elif gov_results["breaker_final_state"] == "HALF_OPEN":
    st.warning("⚠️ CIRCUIT BREAKER HALF-OPEN: Sending canary requests to check if downstream systems have recovered.")
else:
    st.success("✅ CIRCUIT BREAKER CLOSED: System operating normally with all health checks green.")

st.subheader("📋 Execution Trace Stream")
trace_df = pd.DataFrame(gov_results["trace_log"])
st.dataframe(trace_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("11. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions on Production Reliability")

with st.expander("Q1: Why is an idempotency key critical when an autonomous agent executes API tools?"):
    st.markdown("""
    **Answer**: Autonomous agents execute external side-effects (e.g., initiating bank transfers, issuing refund requests). 
    If a transient network disconnect drops the HTTP response packet after the tool has already executed, the agent runtime 
    or user will retry the action. Without an idempotency key, the external API will execute the mutation a second time, 
    causing duplicate transactions. An idempotency key ensures that repeated requests safely return the cached previous result.
    """)

with st.expander("Q2: How does PostgresSaver differ from MemorySaver in LangGraph, and why is MemorySaver unsuitable for production?"):
    st.markdown("""
    **Answer**: `MemorySaver` stores state snapshots in volatile Python process RAM. If the container restarts or crashes, 
    all thread states and conversation histories are lost permanently. Furthermore, in a multi-pod cluster, a second request 
    routed to a different pod cannot see the in-memory state of the first pod. `PostgresSaver` persists state snapshots to a 
    shared relational database, making states durable across reboots and enabling horizontal scaling across worker pods.
    """)

with st.expander("Q3: What are the 3 states of a Circuit Breaker, and what triggers transitions between them?"):
    st.markdown("""
    **Answer**: 
    1. **CLOSED**: Normal operation. Transitions to OPEN when the observed failure percentage exceeds a threshold (e.g., > 25%).
    2. **OPEN**: The circuit fails fast immediately without invoking the broken dependency. Transitions to HALF-OPEN after a cooldown timer expires.
    3. **HALF-OPEN**: Sends a small number of canary probe requests. If they succeed, it resets to CLOSED; if they fail, it trips back to OPEN.
    """)

with st.expander("Q4: What is State Time-Travel in LangGraph checkpointing, and when is it used in production?"):
    st.markdown("""
    **Answer**: Because checkpointers store an append-only sequence of snapshots for every node transition, engineers can 
    query and load state at any historical checkpoint ID. This enables developers to rewind a failed agent run to the exact step 
    before a hallucination occurred, adjust prompt parameters, and re-execute from that point for root-cause analysis.
    """)

with st.expander("Q5: What is the difference between a rate limiter and a circuit breaker?"):
    st.markdown("""
    **Answer**: A **Rate Limiter** protects systems from overload by capping throughput according to pre-set rate policies 
    (e.g., max 100 requests per minute). A **Circuit Breaker** detects when a downstream service is actively failing and 
    stops sending traffic entirely to allow the failing service to recover.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("**Trap 1 (Using MemorySaver in Clustered Environments)**: Using MemorySaver across multiple Kubernetes pods. Users receive random 404s when requests land on pods that don't have the in-memory session.")
    st.warning("**Trap 2 (Non-Idempotent Tool Actions)**: Allowing payment or email tools to execute without deduplication keys. Network timeouts cause duplicate charges during automated retries.")
with col_tr2:
    st.error("**Trap 3 (Missing Circuit Breakers on External APIs)**: Pounding a failing downstream API with high-frequency retries. This prolongs downstream outages and exhausts local application worker threads.")
    st.warning("**Trap 4 (Unbounded Checkpoint Database Growth)**: Writing state snapshots without retention or archival policies. High-volume chat systems will eventually fill database disks with millions of stale historical checkpoints.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary purpose of an Idempotency-Key?")
        st.caption("Flip for answer:")
        st.info("**Back**: To ensure that identical client retries return cached outcomes without re-executing stateful side-effects.")

    with st.container(border=True):
        st.markdown("**Front**: When does a circuit breaker transition to HALF-OPEN?")
        st.caption("Flip for answer:")
        st.info("**Back**: After a cooldown period expires, allowing a small canary probe to test downstream recovery.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why is PostgresSaver required for horizontal scaling in LangGraph?")
        st.caption("Flip for answer:")
        st.info("**Back**: Because it stores thread snapshots in a shared database accessible by any container pod in the cluster.")

    with st.container(border=True):
        st.markdown("**Front**: What is a Dead-Letter Queue (DLQ)?")
        st.caption("Flip for answer:")
        st.info("**Back**: A buffer where unrecoverable requests and traces are routed for offline triage and post-mortem analysis.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Production Reliability in LangChain:")

c1 = st.checkbox("I understand how idempotency keys prevent duplicate side-effects during network retries.", value=True)
c2 = st.checkbox("I know how to configure durable checkpointers (PostgresSaver) to persist thread state across pod restarts.", value=True)
c3 = st.checkbox("I understand the 3 states of a Circuit Breaker (CLOSED, OPEN, HALF-OPEN) and their transition criteria.", value=True)
c4 = st.checkbox("I can implement durable asynchronous queues to support long-running workflows.", value=True)
c5 = st.checkbox("I know how to utilize time-travel debugging to inspect and replay historical checkpoints.", value=True)
c6 = st.checkbox("I understand how token-bucket rate limiting prevents tenant quota exhaustion.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Production Reliability, state durability, and mission-critical resilience architectures in LangChain.")
