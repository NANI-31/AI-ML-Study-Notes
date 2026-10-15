import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="25.5 Production Reliability - LangGraph Curriculum",
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
    .badge-rel {
        background-color: #312E81;
        color: #A5B4FC;
        border: 1px solid #6366F1;
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
class ProductionReliabilitySimulator:
    """
    Simulates production resiliency patterns:
    - Circuit Breaker state machine (CLOSED, OPEN, HALF_OPEN)
    - Retry backoff intervals with jitter
    - Cost budgets and alert threshold monitoring
    """
    def __init__(self):
        self.circuit_state = "CLOSED"
        self.consecutive_failures = 0
        self.failure_threshold = 3
        self.total_cost_usd = 42.50
        self.cost_cap_usd = 100.00

    def trigger_request(self, force_failure: bool = False) -> Dict[str, Any]:
        if self.circuit_state == "OPEN":
            return {
                "status": "CIRCUIT_OPEN_FAST_FAIL",
                "circuit_state": "OPEN",
                "message": "Circuit breaker is OPEN. Fast-failing immediately to protect downstream service."
            }

        if force_failure:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.circuit_state = "OPEN"
                msg = f"Failure threshold reached ({self.consecutive_failures}). Circuit tripped to OPEN!"
            else:
                msg = f"Request failed. Failure count: {self.consecutive_failures}/{self.failure_threshold}"
            return {
                "status": "FAILED",
                "circuit_state": self.circuit_state,
                "message": msg
            }
        else:
            if self.circuit_state == "HALF_OPEN":
                self.circuit_state = "CLOSED"
                self.consecutive_failures = 0
                msg = "Probe request succeeded! Circuit reset to CLOSED."
            else:
                self.consecutive_failures = 0
                msg = "Request processed successfully."
            return {
                "status": "SUCCESS",
                "circuit_state": self.circuit_state,
                "message": msg
            }

    def reset_circuit(self) -> None:
        self.circuit_state = "CLOSED"
        self.consecutive_failures = 0

    def trip_half_open(self) -> None:
        self.circuit_state = "HALF_OPEN"

reliab_sim = ProductionReliabilitySimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Resilience Patterns, Chaos Engineering, and Fault Recovery Lifecycle", divider="rainbow")

st.markdown("""
Production reliability transforms autonomous agents from fragile prototypes into mission-critical enterprise systems. 
Because agents depend on probabilistic external APIs, third-party LLM vendors, and distributed network tools, 
system designs must embrace failures as inevitable. 
A reliable LangGraph architecture integrates defense-in-depth reliability mechanisms: exponential retries with jitter, 
hard per-node execution timeouts, circuit breakers preventing cascading outages, checkpoint-based fault recovery, 
and real-time cost and telemetry alerting.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ReliabilityBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_ingress {
            label="1. Ingress Rate Limiting & Cost Gateways";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            req_in [label="Incoming Request", fillcolor="#0F172A"];
            rate_limiter [label="Token Bucket Rate Limiter\\nEnforces RPM & TPM Quotas", fillcolor="#1E293B"];
            cost_gate [label="Cost Cap Enforcer\\nVerifies Org Budget < Daily Ceiling", fillcolor="#1E293B"];
        }

        subgraph cluster_execution {
            label="2. Resilient Superstep Execution Runtime";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            node_exec [label="Node Execution Boundary\\nEnforces async Timeout (30s)", fillcolor="#312E81"];
            retry_policy [label="Retry Policy Engine\\nExponential Backoff (1s, 2s, 4s) + Jitter", fillcolor="#1E293B"];
            circuit_breaker [label="Circuit Breaker Gate\\nTrips OPEN on 3 Consecutive Failures", fillcolor="#831843", fontcolor="#FBCFE8"];
            model_target [label="External LLM / Tool Provider\\nProne to 429s, 503s & Timeouts", fillcolor="#0F172A"];
        }

        subgraph cluster_recovery {
            label="3. Fault Recovery & Observability";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            fallback_node [label="Graceful Fallback Node\\nReturns Degraded Mode Response", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            checkpoint_store [label="Postgres Checkpoint Store\\nPreserves State for Resume After Crash", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            alert_stream [label="Alerting & Metrics Engine\\nPrometheus, PagerDuty, Slack Webhooks", fillcolor="#0F172A"];
        }

        req_in -> rate_limiter;
        rate_limiter -> cost_gate;
        cost_gate -> node_exec;

        node_exec -> circuit_breaker;
        circuit_breaker -> model_target [label="Normal (CLOSED)"];
        model_target -> retry_policy [label="Transient 429/500"];
        retry_policy -> node_exec [label="Retry with Backoff"];

        circuit_breaker -> fallback_node [label="Fast-Fail (OPEN)"];
        node_exec -> checkpoint_store [label="Commit Step"];
        node_exec -> alert_stream [label="Emit SLI Metrics"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Retries
# ==============================================================================
st.header("2. Retries: Exponential Backoff, Jitter, and Idempotent Node Re-Execution", divider="rainbow")

st.markdown("""
Network blips and provider rate limits (HTTP 429) are normal occurrences in distributed AI pipelines. 
Naive immediate retries cause "thundering herd" problems, exacerbating provider load.

### Production Retry Best Practices:
1. **Exponential Backoff**: Increase wait time exponentially after each attempt (e.g., 1s, 2s, 4s, 8s).
2. **Full Jitter**: Add randomized noise to backoff delays to prevent synchronized retries across concurrent pods:
   `wait_time = random_uniform(0, min(max_backoff, base_backoff * (2 ** attempt)))`
3. **Idempotency Guarantees**: Retried nodes must be idempotent; executing the node twice should not result 
   in double database inserts or duplicate financial transactions.
""")

st.code('''from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
async def resilient_llm_call(prompt: str) -> str:
    # Retries up to 4 times with randomized exponential jitter
    return await llm_client.ainvoke(prompt)
''', language="python")

# ==============================================================================
# SECTION 3: Timeouts
# ==============================================================================
st.header("3. Timeouts: Per-Node Execution Deadlines, Global Graph Timeouts, and Async Cancellation", divider="rainbow")

st.markdown("""
An agent step should never hang indefinitely awaiting a stuck tool or slow model response. 
Timeouts must be enforced at both the individual node level and the global graph invocation level.

### Multi-Tier Timeout Architecture:
- **Tool Timeout (5-15s)**: External APIs and database queries must carry aggressive timeouts.
- **Node Timeout (30-60s)**: An entire node execution (including reasoning and tool dispatch) must terminate if stalled.
- **Global Graph Timeout (120-300s)**: Overall `app.invoke()` must be wrapped in `asyncio.wait_for` to guarantee a hard ceiling on request duration.
""")

st.code('''import asyncio

# Enforcing a hard 45-second deadline on full graph execution
try:
    result = await asyncio.wait_for(
        app.ainvoke({"messages": [HumanMessage(content="Audit accounts")]}, config),
        timeout=45.0
    )
except asyncio.TimeoutError:
    # Graceful timeout fallback
    result = {"messages": [AIMessage(content="Operation timed out. Progress saved to checkpoint.")]}
''', language="python")

# ==============================================================================
# SECTION 4: Rate Limits
# ==============================================================================
st.header("4. Rate Limits: Token Bucket Algorithms, Provider Concurrency Throttling, and 429 Interception", divider="rainbow")

st.markdown("""
Foundation model providers enforce strict rate limits based on **Requests Per Minute (RPM)** and **Tokens Per Minute (TPM)**. 
Production LangGraph systems utilize rate limiting middleware to smooth traffic spikes.

### Strategies:
- **Client-Side Token Bucket**: Maintain local token bucket rate limiters (e.g., using Redis) to queue outgoing requests before exceeding vendor limits.
- **Concurrency Semaphores**: Bound simultaneous active model invocations to a fixed semaphore pool (e.g., `asyncio.Semaphore(20)`).
- **Proactive TPM Tracking**: Compute token counts using `tiktoken` before dispatching requests, deferring execution if the current rolling 60-second window is saturated.
""")

# ==============================================================================
# SECTION 5: Circuit Breakers
# ==============================================================================
st.header("5. Circuit Breakers: State Transitions, Failure Thresholds, and Automatic Reset", divider="rainbow")

st.markdown("""
A **Circuit Breaker** protects systems from cascading failures when an external dependency (such as an LLM provider or database) 
suffers an outage. Instead of allowing hundreds of requests to pile up and wait for timeouts, the circuit breaker "trips" 
and immediately returns fallback responses.
""")

with st.expander("🔍 View Circuit Breaker State Transition Architecture", expanded=True):
    cb_dot = """
    digraph CircuitBreakerState {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        closed [label="CLOSED (Normal Operation)\\nAll requests forwarded to LLM/Tool\\nSuccess resets failure counter", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        open_state [label="OPEN (Tripped Failure)\\nRequests immediately fast-fail\\nNo traffic sent to failing service", fillcolor="#831843", fontcolor="#FBCFE8"];
        half_open [label="HALF-OPEN (Probe Mode)\\nAllows 1 test request through\\nChecks if service has recovered", fillcolor="#312E81"];

        closed -> open_state [label="Consecutive Failures >= Threshold (e.g., 5)"];
        open_state -> half_open [label="Cooldown Timer Expires (e.g., 60s)"];
        half_open -> closed [label="Probe Request Succeeds"];
        half_open -> open_state [label="Probe Request Fails"];
    }
    """
    st.graphviz_chart(cb_dot, use_container_width=True)

# ==============================================================================
# SECTION 6: Failure Recovery
# ==============================================================================
st.header("6. Failure Recovery: Fallback Branches, Partial Graph Recovery, and Checkpoint Resurrection", divider="rainbow")

st.markdown("""
When a node suffers an unrecoverable failure, robust LangGraph graphs avoid aborting the entire user session.

### Recovery Mechanisms:
1. **Conditional Fallback Branches**: Route exceptions into a dedicated fallback node that returns a degraded answer or provides alternative suggestions.
2. **Partial Graph Recovery**: If Node 4 of 6 crashes, the previous 3 supersteps remain safely committed in PostgreSQL. 
   When the service recovers, the worker reads the latest valid checkpoint and resumes from Node 4 without re-running Nodes 1-3.
3. **Pending Writes Clearance**: Clear corrupted pending writes using `app.update_state()` to restore a healthy execution branch.
""")

# ==============================================================================
# SECTION 7: Monitoring
# ==============================================================================
st.header("7. Monitoring: Prometheus Metrics, Golden Signals, and Grafana Dashboards", divider="rainbow")

st.markdown("""
Production LangGraph microservices must export standard **Golden Signals** (Latency, Traffic, Errors, and Saturation):

- **Prometheus Metrics**:
  - `langgraph_superstep_duration_seconds`: Histogram of superstep execution durations.
  - `langgraph_llm_tokens_total`: Counter partitioned by `provider`, `model`, and `token_type` (prompt vs completion).
  - `langgraph_node_errors_total`: Counter of exceptions partitioned by node name and error type.
  - `langgraph_active_streaming_connections`: Gauge tracking current live SSE clients.
""")

# ==============================================================================
# SECTION 8: Alerting
# ==============================================================================
st.header("8. Alerting: PagerDuty Integration, SLI/SLO Breach Detection, and Anomaly Thresholds", divider="rainbow")

st.markdown("""
Alerting policies notify on-call engineers before user experience degrades severely.

### Key SLO & Alert Definitions:
- **Error Budget Alert**: Trigger PagerDuty if error rate on primary agent endpoints exceeds 1.5% over a rolling 15-minute window.
- **p95 Latency Alert**: Alert if p95 response time exceeds 8 seconds over 10 minutes.
- **Provider Outage Alert**: Alert if 5 consecutive circuit breaker trips occur across multiple tenant threads.
""")

# ==============================================================================
# SECTION 9: Cost Monitoring
# ==============================================================================
st.header("9. Cost Monitoring: Per-Tenant Token Attribution, Cost Alarms, and Daily Spend Caps", divider="rainbow")

st.markdown("""
Autonomous agent loops pose significant financial risks if a buggy conditional edge enters an uncontrolled loop. 
Strict cost monitoring prevents runaway billing:

- **Per-Tenant Quotas**: Maintain daily and monthly spend budgets per customer organization.
- **Hard Spend Caps**: If a tenant's cumulative spend reaches 95% of their monthly budget, trigger an interrupt or refuse further runs.
- **Anomaly Detection**: Flag sudden spikes in token consumption (e.g., a single thread consuming over 100,000 tokens in under 5 minutes).
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Enterprise Resiliency Wrapper with Circuit Breaker and Retry Policies", divider="rainbow")

st.markdown("""
Below is a complete enterprise resiliency wrapper providing exponential retries, circuit breaker state tracking, 
and automatic fallback execution for LangGraph agents.
""")

st.code('''from typing import Callable, Any, Dict, Optional
import time

class ResilientGraphExecutor:
    """
    Enterprise execution harness wrapping LangGraph calls with
    circuit breaking, exponential retries, and fallback handling.
    """
    def __init__(self, compiled_graph, max_retries: int = 3, failure_threshold: int = 5, cooldown_sec: float = 60.0):
        self.graph = compiled_graph
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = 0.0

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        
        # Check circuit cooldown
        if self.state == "OPEN":
            if now - self.last_failure_time > self.cooldown_sec:
                self.state = "HALF_OPEN"
            else:
                return {"error": "Circuit breaker OPEN. System degraded.", "fallback_mode": True}

        # Attempt invocation with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.graph.invoke(inputs, config)
                # Success resets circuit
                self.state = "CLOSED"
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = now
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    
                if attempt == self.max_retries:
                    # Execute fallback logic
                    return {
                        "error": f"Execution failed after {self.max_retries} attempts: {str(e)}",
                        "fallback_mode": True,
                        "thread_id": config.get("configurable", {}).get("thread_id")
                    }
                time.sleep(0.5 * (2 ** (attempt - 1)))
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Fault Injection Simulator, Circuit Breaker Sandbox & Cost Monitor", divider="rainbow")

st.markdown("""
Test circuit breaker trip states, simulate transient failures, and audit cost spend limits in real time.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("Circuit Breaker & Chaos Simulator")
    
    st.markdown(f"**Current Circuit State:** `{reliab_sim.circuit_state}`")
    st.markdown(f"**Consecutive Failures:** `{reliab_sim.consecutive_failures} / {reliab_sim.failure_threshold}`")
    
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("Simulate Success", type="primary"):
            res = reliab_sim.trigger_request(force_failure=False)
            st.rerun()
    with btn2:
        if st.button("Inject Fault (Failure)"):
            res = reliab_sim.trigger_request(force_failure=True)
            st.rerun()
    with btn3:
        if st.button("Reset Circuit"):
            reliab_sim.reset_circuit()
            st.rerun()

    if reliab_sim.circuit_state == "CLOSED":
        st.success("Circuit CLOSED: System operating under healthy baseline conditions.")
    elif reliab_sim.circuit_state == "OPEN":
        st.error("Circuit OPEN: Downstream service failure threshold exceeded. Fast-failing active traffic!")
    else:
        st.warning("Circuit HALF-OPEN: Testing recovery with single probe request.")

with col_wb2:
    st.subheader("Tenant Cost Budget & Quota Guard")
    current_spend = st.slider("Simulated Monthly Spend (USD)", min_value=10.0, max_value=150.0, value=float(reliab_sim.total_cost_usd), step=5.0)
    budget_cap = 100.00
    
    pct_used = (current_spend / budget_cap) * 100
    st.metric("Budget Utilization", f"{pct_used:.1f}%", f"{current_spend:.2f} / {budget_cap:.2f} USD")
    
    if current_spend > budget_cap:
        st.error("HARD CAP EXCEEDED: Autonomous agent loop suspended. Human financial approval required to continue.")
    elif current_spend > 0.8 * budget_cap:
        st.warning("Cost Warning: Approaching 80% monthly spend quota threshold.")
    else:
        st.success("Cost Healthy: Spend is well within allocated monthly quotas.")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: Why must backoff retries incorporate random jitter in high-concurrency agent systems?"):
    st.markdown("""
    **Answer:**  
    Without jitter, multiple concurrent requests that fail at the same instant (e.g., during a brief API rate-limit spike) 
    will retry at the exact same synchronized intervals (1s, 2s, 4s), producing repeated spikes that overwhelm the provider repeatedly. 
    Random jitter scatters the retries across a time window, smoothing the traffic curve.
    """)

with st.expander("Q2: How does a Circuit Breaker differ from standard retries?"):
    st.markdown("""
    **Answer:**  
    Retries attempt to resolve transient, short-lived errors on individual requests. Circuit breakers address prolonged outages 
    by intercepting traffic at the system level; once failures exceed a threshold, the circuit trips to `OPEN`, immediately fast-failing 
    all subsequent calls to protect the downstream system and avoid wasting resources on guaranteed timeouts.
    """)

with st.expander("Q3: How does LangGraph checkpointing facilitate partial workflow failure recovery?"):
    st.markdown("""
    **Answer:**  
    Because LangGraph persists the state after every superstep, an unhandled exception at Step 5 does not destroy the progress of Steps 1-4. 
    Once the underlying issue is resolved (or when a fallback node is engaged), the application reloads the checkpoint from Step 4 
    and continues execution without repeating expensive preceding steps.
    """)

with st.expander("Q4: What is the most effective safeguard against runaway infinite agent loops?"):
    st.markdown("""
    **Answer:**  
    Enforce a strict `recursion_limit` in the `RunnableConfig` (e.g., `config={"recursion_limit": 25}`). 
    If the graph cycles beyond this superstep limit, LangGraph automatically halts execution with a `GraphRecursionError`, 
    preventing infinite token consumption and billing exhaustion.
    """)

with st.expander("Q5: What are the primary Golden Signals to monitor on LangGraph agent services?"):
    st.markdown("""
    **Answer:**  
    1. **Latency**: Superstep duration and Time to First Token (TTFT).  
    2. **Traffic**: Active streaming SSE connections and supersteps per minute.  
    3. **Errors**: Exception rate, tool failure rate, and circuit breaker trip events.  
    4. **Saturation**: PostgreSQL checkpointer connection pool utilization and worker queue depth.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Retrying Non-Transient 4xx Client Errors**  
Retrying client errors such as `400 Bad Request` or `401 Unauthorized` with exponential backoff wastes time and compute, 
as client payload errors will never succeed on retry. Only retry transient codes like `429 Too Many Requests`, `502 Bad Gateway`, or `503 Service Unavailable`.
""")

st.error("""
**Trap 2: Omitting recursion_limit in Cyclic Workflows**  
Failing to specify or tune `recursion_limit` allows poorly prompt-engineered agents to enter infinite critique-regeneration cycles, 
burning through thousands of dollars in token usage within hours.
""")

st.warning("""
**Trap 3: Masking Errors Inside Fallback Nodes Without Alerting**  
Catching exceptions and returning fallback messages without incrementing error counters or logging to Prometheus hides production degradation 
from monitoring dashboards until users report issues.
""")

st.warning("""
**Trap 4: Missing Timeouts on Tool Database Queries**  
Omitting statement timeouts on SQL tools allows long-running or deadlocked queries to hang the node indefinitely, 
holding worker threads and database pool connections hostage.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Circuit Breaker</h4>
        <p>A stability pattern that fast-fails requests when an external service is down, allowing the failing service time to recover.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Exponential Backoff + Jitter</h4>
        <p>A retry strategy exponentially increasing wait times while introducing random variation to prevent synchronized thundering herds.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>recursion_limit</h4>
        <p>The maximum number of superstep execution cycles permitted per invocation before LangGraph terminates execution with an error.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Checkpoint Resurrection</h4>
        <p>Resuming interrupted or failed agent workflows from the last committed valid superstep snapshot without re-running prior nodes.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to implement exponential backoff retries with randomized jitter.", value=True)
chk_2 = st.checkbox("I understand how to configure per-node deadlines and global graph execution timeouts.", value=True)
chk_3 = st.checkbox("I can implement circuit breaker state transitions (CLOSED, OPEN, HALF-OPEN) to prevent cascading failures.", value=True)
chk_4 = st.checkbox("I know how to recover from partial graph failures using persistent checkpoints.", value=True)
chk_5 = st.checkbox("I understand how to monitor Golden Signals (latency, errors, tokens) with Prometheus and Grafana.", value=True)
chk_6 = st.checkbox("I can enforce per-tenant token budgets and spend caps to eliminate runaway billing risks.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Production reliability and resilience engineering mastery confirmed! Chapters 23, 24, and 25 complete!")
