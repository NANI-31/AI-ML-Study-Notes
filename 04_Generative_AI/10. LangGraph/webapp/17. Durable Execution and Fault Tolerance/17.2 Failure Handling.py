import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="17.2 Failure Handling - LangGraph Curriculum",
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
    .badge-error {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-retry {
        background-color: #382B14;
        color: #FBBF24;
        border: 1px solid #D97706;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-ok {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class FailureHandlingSimulator:
    """
    Simulates production failure scenarios and resilience mechanics in LangGraph:
    - Node exception traps & dead-letter routing
    - Tool-level execution sandbox (handle_tool_errors)
    - Exponential backoff with jitter
    - Model fallback cascades on context window exhaustion or rate limits
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.consecutive_failures = 0
        self.total_tokens_used = 0

    def execute_node_with_retry(
        self,
        node_name: str,
        fail_rate: float,
        max_retries: int,
        base_delay_sec: float,
        jitter: bool
    ) -> Dict[str, Any]:
        attempts = 0
        logs = []
        success = False

        while attempts <= max_retries:
            attempts += 1
            roll = np.random.uniform(0.0, 1.0)
            if roll >= fail_rate:
                success = True
                logs.append({
                    "attempt": attempts,
                    "status": "SUCCESS",
                    "latency_sec": round(0.1 + np.random.uniform(0.05, 0.15), 3),
                    "error": None
                })
                break
            else:
                backoff = base_delay_sec * (2 ** (attempts - 1))
                if jitter:
                    backoff = backoff * (0.5 + np.random.uniform(0.0, 0.5))
                backoff = round(backoff, 3)
                logs.append({
                    "attempt": attempts,
                    "status": "RETRY_TRIGGERED" if attempts <= max_retries else "TERMINAL_FAILURE",
                    "latency_sec": backoff,
                    "error": f"SimulatedConnectionTimeout in {node_name}"
                })

        return {
            "node": node_name,
            "success": success,
            "total_attempts": attempts,
            "logs": logs
        }

    def simulate_tool_execution(
        self,
        tool_name: str,
        sandboxed: bool,
        crash_simulation: bool
    ) -> Dict[str, Any]:
        if not crash_simulation:
            return {
                "tool": tool_name,
                "status": "OK",
                "result": f"Data retrieved successfully from {tool_name}",
                "graph_crashed": False
            }

        if sandboxed:
            return {
                "tool": tool_name,
                "status": "HANDLED_ERROR",
                "result": f"ToolException caught by ToolNode: '{tool_name} backend timed out'. Passed back to LLM as ToolMessage.",
                "graph_crashed": False
            }
        else:
            return {
                "tool": tool_name,
                "status": "UNHANDLED_EXCEPTION",
                "result": f"Traceback: KeyError in {tool_name}. Graph aborted immediately.",
                "graph_crashed": True
            }

    def simulate_model_cascade(
        self,
        primary_model: str,
        fallback_model: str,
        primary_context_limit: int,
        prompt_tokens: int
    ) -> Dict[str, Any]:
        if prompt_tokens <= primary_context_limit:
            return {
                "active_model": primary_model,
                "used_fallback": False,
                "status": "OK",
                "message": f"Prompt ({prompt_tokens} tokens) processed successfully by {primary_model}."
            }
        else:
            return {
                "active_model": fallback_model,
                "used_fallback": True,
                "status": "FALLBACK_ENGAGED",
                "message": f"Primary {primary_model} rejected context ({prompt_tokens} > {primary_context_limit} tokens). Automatically routed to {fallback_model} (128k window)."
            }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Failure Topologies and Cascade Recovery in LangGraph", divider="rainbow")

st.markdown("""
In distributed AI agent systems, failures are guaranteed occurrences rather than exceptional anomalies.
A resilient LangGraph architecture categorizes failures into distinct engineering tiers:
- **Node-Level Failures**: Transient network outages, unhandled code bugs, or resource starvation within node functions.
- **Tool-Level Failures**: External API breaks, schema validation errors, or non-responsive 3rd-party services called by agents.
- **Model-Level Failures**: Context window overflow (HTTP 400), rate limit throttling (HTTP 429), or complete provider downtime (HTTP 503).
- **Execution Deadlines**: Runaway loops or slow API queries that exceed designated SLA timeouts.

The blueprint below illustrates how LangGraph isolates exceptions at each layer and routes execution through retries, fallbacks, and dead-letter channels without crashing the application thread.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FailureHandlingTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_ingress {
            label="1. Ingress & Superstep Dispatch";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            client_req [label="User / External Request\\n(Thread Configured)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            step_dispatch [label="LangGraph Engine\\nSuperstep Coordinator", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_model_tier {
            label="2. Primary & Fallback Model Execution Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            primary_llm [label="Primary Model (e.g. GPT-4o)\\nTimeout: 15s", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            model_eval [label="Model Exception Filter\\n429 RateLimit / 400 Context", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            fallback_llm [label="Fallback Model (Claude 3.5 Sonnet)\\nLarge Window / Secondary Host", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_tool_tier {
            label="3. Tool Sandboxing & ToolNode Resilience";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            tool_dispatch [label="ToolNode Sandbox\\nhandle_tool_errors=True", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            tool_api [label="External 3rd-Party APIs\\n(Weather, SQL, Search)", fillcolor="#18181B", fontcolor="#E4E4E7", color="#71717A"];
            tool_msg_err [label="Synthetic ToolMessage\\nstatus='error' -> Returned to State", fillcolor="#701A75", fontcolor="#F5D0FE", color="#E879F9"];
        }

        subgraph cluster_recovery {
            label="4. Fault Tolerance & Dead-Letter Channel";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            retry_policy [label="Exponential Backoff Policy\\nAttempts: 3 | Jitter: Uniform", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            checkpoint_store [label="Durable Checkpointer\\n(PostgresSaver DB)", fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            dead_letter [label="Dead-Letter Queue / Alert\\nHuman Escalation Required", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
        }

        client_req -> step_dispatch;
        step_dispatch -> primary_llm [label="Invokes Node"];
        primary_llm -> model_eval [label="Throws Exception", color="#EF4444", style=dashed];
        primary_llm -> tool_dispatch [label="Tool Calls Generated", color="#34D399"];
        model_eval -> fallback_llm [label="Fallback Route", color="#38BDF8"];
        fallback_llm -> tool_dispatch [label="Valid Tool Calls", color="#34D399"];
        model_eval -> retry_policy [label="Transient 503", color="#FBBF24"];
        retry_policy -> primary_llm [label="Retry with Backoff", color="#FBBF24", style=dashed];

        tool_dispatch -> tool_api [label="HTTP Request"];
        tool_api -> tool_msg_err [label="Network / Schema Crash", color="#EF4444", style=dashed];
        tool_msg_err -> checkpoint_store [label="State Preserved"];
        checkpoint_store -> step_dispatch [label="Loop Continues with Error Feedback", color="#38BDF8"];

        retry_policy -> dead_letter [label="Max Retries Exhausted", color="#EF4444"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: NODE FAILURES
# ==============================================================================
st.header("2. Node Failures: Isolation, Exception Trapping, and Dead-Letter Routing", divider="rainbow")

st.markdown("""
When a node throws an unhandled Python exception during execution, LangGraph halts the active superstep.
However, because of the Pregel architecture:
1. **Prior Supersteps are Preserved**: All state updates committed up to the beginning of the failing superstep remain persisted in the checkpointer.
2. **Current Step Writes are Discarded**: No partial, corrupted state mutations from the crashing node pollute the graph state.
3. **Dead-Letter Routing**: Rather than crashing the entire process, architects wrap nodes with structured exception guards or configure error branches in conditional edges.
""")

col_n1, col_n2 = st.columns(2)
with col_n1:
    st.markdown("#### The Dangerous Pattern: Uncaught Node Crashes")
    st.code('''# ANTI-PATTERN: Direct uncaught network calls in node
def fragile_fetch_node(state: State) -> dict:
    # If the network drops, the entire thread aborts with unhandled exception!
    resp = requests.get("https://internal.crm.api/users", timeout=5)
    data = resp.json()
    return {"user_records": data}''', language="python")

with col_n2:
    st.markdown("#### The Production Pattern: Exception Trapping & Routing")
    st.code('''# PRODUCTION PATTERN: Trapping errors into state channels
def resilient_fetch_node(state: State) -> dict:
    try:
        resp = requests.get("https://internal.crm.api/users", timeout=5)
        resp.raise_for_status()
        return {"user_records": resp.json(), "error_state": None}
    except Exception as exc:
        # State records failure; conditional edge routes to recovery
        return {
            "error_state": {"node": "fetch_user", "message": str(exc)},
            "failure_count": state.get("failure_count", 0) + 1
        }''', language="python")

st.info("💡 **Architectural Principle**: A failing node should yield error metadata into the state graph instead of bubbling raw exceptions to the runtime host, enabling downstream routing nodes to make autonomous recovery decisions.")

# ==============================================================================
# SECTION 3: TOOL FAILURES
# ==============================================================================
st.header("3. Tool Failures: Sandboxed Execution and handle_tool_errors Mechanics", divider="rainbow")

st.markdown("""
Agentic applications delegate work to third-party tools (databases, scrapers, search engines).
If a tool crashes due to invalid parameters or external server errors, throwing a Python exception terminates the agent's reasoning loop.
LangGraph provides the built-in `ToolNode` with configurable `handle_tool_errors` logic.
""")

st.markdown("#### How `handle_tool_errors` Works Internally")
st.markdown("""
- **When `handle_tool_errors=True`**: LangGraph catches any standard exception thrown by the tool function.
- It serializes the exception string into a synthetic `ToolMessage` with `status="error"`.
- This `ToolMessage` is injected into the conversation history channel.
- On the next superstep, the LLM receives the error message, learns *why* the tool call failed, and can autonomously correct its parameters or choose an alternate strategy!
""")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.code('''from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def query_weather_sensor(zip_code: str) -> str:
    """Fetch live temperature readings from physical sensors."""
    if len(zip_code) != 5:
        raise ValueError(f"Zip code '{zip_code}' must be exactly 5 digits!")
    return "72F and sunny"

# Production ToolNode with custom error transformer
def custom_tool_error_handler(error: Exception) -> str:
    return f"TOOL EXECUTION ERROR: {type(error).__name__} - {str(error)}. Please refine arguments."

tool_node = ToolNode(
    tools=[query_weather_sensor],
    handle_tool_errors=custom_tool_error_handler
)''', language="python")

with col_t2:
    st.markdown("""
    **Lifecycle Comparison Table**:
    """)
    df_tools = pd.DataFrame({
        "Behavior": ["Exception Action", "Graph State", "Agent Recovery", "Production Suitability"],
        "Default (handle_tool_errors=False)": ["Uncaught Python Traceback", "Aborted immediately", "Impossible without restart", "Unsafe"],
        "Configured (handle_tool_errors=True)": ["Captured as ToolMessage", "Committed to Thread", "LLM sees error & re-prompts", "Mandatory"]
    })
    st.dataframe(df_tools, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 4: API FAILURES
# ==============================================================================
st.header("4. API Failures: HTTP 429, 500, 503 Outage Resilience and Circuit Breakers", divider="rainbow")

st.markdown("""
External LLM and SaaS API outages fall into three standard categories:
1. **HTTP 429 (Rate Limit Exceeded)**: The quota or token-per-minute threshold was breached. Requires exponential backoff with randomized jitter.
2. **HTTP 500 / 502 / 503 (Server Unavailable)**: Remote host malfunction. Requires retry policies or automatic failover to a warm mirror.
3. **Cascading Saturation**: If a service is failing, continuing to bombard it with retries creates a thundering herd. A **Circuit Breaker** must trip to OPEN, failing fast without hitting the remote API.
""")

with st.expander("🔍 View Circuit Breaker State Transition Blueprint", expanded=False):
    cb_dot = """
    digraph CircuitBreakerState {
        bgcolor="#0B1120";
        rankdir=LR;
        node [shape=circle, style="filled", fontname="Segoe UI, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1"];

        closed [label="CLOSED\\n(Normal Ops)", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        open [label="OPEN\\n(Fast-Fail)", fillcolor="#3B1A1A", fontcolor="#F87171", color="#DC2626"];
        half_open [label="HALF-OPEN\\n(Probe Canary)", fillcolor="#382B14", fontcolor="#FBBF24", color="#D97706"];

        closed -> open [label="Failure Threshold Exceeded", color="#EF4444"];
        open -> half_open [label="Cooldown Timer Expired", color="#FBBF24"];
        half_open -> closed [label="Probe Succeeds", color="#34D399"];
        half_open -> open [label="Probe Fails", color="#EF4444"];
    }
    """
    st.graphviz_chart(cb_dot, use_container_width=True)

st.markdown("""
```python
# Circuit Breaker Decorator for LangGraph Node Functions
class NodeCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_sec: float = 30.0):
        self.threshold = failure_threshold
        self.cooldown = cooldown_sec
        self.failures = 0
        self.state = "CLOSED"
        self.last_trip_time = 0.0

    def call_api(self, func, *args, **kwargs):
        now = time.time()
        if self.state == "OPEN":
            if now - self.last_trip_time > self.cooldown:
                self.state = "HALF-OPEN"
            else:
                raise RuntimeError("CircuitBreaker OPEN: Fast-failing external call.")
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as exc:
            self.failures += 1
            if self.failures >= self.threshold:
                self.state = "OPEN"
                self.last_trip_time = now
            raise exc
```
""")

# ==============================================================================
# SECTION 5: MODEL FAILURES
# ==============================================================================
st.header("5. Model Failures: Context Limit Exceeded and Secondary Provider Cascades", divider="rainbow")

st.markdown("""
Large Language Models exhibit unique failure modes distinct from standard microservices:
- **Context Window Overflow (`context_length_exceeded`)**: Conversation history outgrows the model's maximum input tokens (e.g., 8k or 128k).
- **Refusal or Content Filtering**: Safety guardrails reject the prompt or tool output.
- **Provider Outage**: Complete service failure of the primary model provider (e.g., OpenAI outage).

LangGraph addresses this with **Model Cascades** using LangChain's `.with_fallbacks()` protocol directly inside graph nodes.
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Primary high-speed model
primary_model = ChatOpenAI(model="gpt-4o", max_retries=2, request_timeout=15.0)

# Secondary large-context backup model across independent infrastructure
backup_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_retries=2, request_timeout=25.0)

# Resilient composite runnable
resilient_model = primary_model.with_fallbacks([backup_model])

def agent_reasoning_node(state: State) -> dict:
    # If primary_model throws RateLimitError, ContextExceeded, or 503,
    # fallback_model executes seamlessly without graph termination!
    ai_message = resilient_model.invoke(state["messages"])
    return {"messages": [ai_message]}''', language="python")

st.markdown("""
> **Key Architecture Insight**: Using `with_fallbacks` ensures that cross-provider redundancy operates transparently at the node layer, while thread checkpointers record only the final successful output message.
""")

# ==============================================================================
# SECTION 6: TIMEOUT HANDLING
# ==============================================================================
st.header("6. Timeout Handling: Request Deadlines and Async Cancellation Guardrails", divider="rainbow")

st.markdown("""
In conversational and batch systems, a hanging network socket can freeze execution indefinitely.
LangGraph supports timeout control at two distinct granularities:
1. **Invocation-Level Timeout**: Setting an execution budget on the graph compilation runner via `asyncio.wait_for`.
2. **Node-Level Client Deadlines**: Passing specific `timeout` configurations to underlying clients within node implementations.
""")

col_to1, col_to2 = st.columns(2)
with col_to1:
    st.markdown("#### Async Graph Invocation Timeout")
    st.code('''import asyncio

async def run_with_deadline(app, inputs, config, deadline_seconds=10.0):
    try:
        return await asyncio.wait_for(
            app.ainvoke(inputs, config=config),
            timeout=deadline_seconds
        )
    except asyncio.TimeoutError:
        # State up to last checkpoint remains safe in DB
        print(f"Workflow exceeded deadline of {deadline_seconds}s!")
        # Can inspect thread state to see last committed superstep
        snapshot = app.get_state(config)
        print("Last completed step:", snapshot.next)
        return {"status": "TIMEOUT", "last_snapshot": snapshot}''', language="python")

with col_to2:
    st.markdown("#### Key Production Rules for Timeouts")
    st.markdown("""
    - **Fast Tool Deadlines**: External tools should have strict socket timeouts (e.g. 5.0 seconds).
    - **Graceful Cancellation**: When an async graph is cancelled, pending writes are discarded cleanly.
    - **Inspect via State Snapshot**: After a timeout aborts an invocation, `app.get_state(config)` accurately reveals which node was running when the deadline fired.
    """)

# ==============================================================================
# SECTION 7: RETRY STRATEGIES
# ==============================================================================
st.header("7. Retry Strategies: Exponential Backoff, Jitter, and Retry Policies", divider="rainbow")

st.markdown("""
Retrying without backoff guarantees that an overloaded downstream API will fail completely.
LangGraph nodes support built-in retry policies configured directly on graph nodes via `RetryPolicy`.
""")

st.code('''from langgraph.types import RetryPolicy

# Configure formal node retry policy in LangGraph
robust_retry_policy = RetryPolicy(
    initial_interval=1.0,      # Start with 1.0 second delay
    backoff_factor=2.0,        # Double delay on each failure: 1s -> 2s -> 4s
    max_interval=30.0,         # Cap maximum sleep delay at 30 seconds
    max_attempts=4,            # Attempt up to 4 total times (1 initial + 3 retries)
    jitter=True,               # Add uniform jitter: prevents synchronized thundering herds
    retry_on=(ConnectionError, TimeoutError) # Only retry on transient transport errors
)

# Attach policy to node registration
# builder.add_node("crm_sync", sync_crm_node, retry_policy=robust_retry_policy)''', language="python")

st.markdown("""
**The Mathematical Formula for Backoff with Jitter**:
`delay = min(max_interval, initial_interval * (backoff_factor ^ (attempt - 1))) * uniform(0.5, 1.5)`
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Tier Resilient Failure Handler Graph", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable, production-grade LangGraph workflow demonstrating:
1. Node-level exception catching
2. Fallback routing via conditional edges
3. Synthetic error messaging
4. Resilient checkpointer persistence
""")

st.code('''from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ResilientState(TypedDict):
    query: str
    data_payload: Optional[dict]
    error_trace: Optional[str]
    retry_count: int
    recovery_strategy: Optional[str]

def primary_data_node(state: ResilientState) -> dict:
    query = state.get("query", "")
    # Simulate transient network inspection
    if "simulate_fail" in query:
        raise ConnectionResetError("Primary gateway dropped connection abruptly.")
    return {"data_payload": {"status": "SUCCESS", "records": [101, 102, 103]}}

def fallback_recovery_node(state: ResilientState) -> dict:
    # Engaged when primary data node fails
    return {
        "recovery_strategy": "READ_REPLICA_CACHE",
        "data_payload": {"status": "CACHED_FALLBACK", "records": [101]},
        "error_trace": "Recovered from primary failure via read replica."
    }

def dead_letter_alert_node(state: ResilientState) -> dict:
    return {
        "recovery_strategy": "MANUAL_ESCALATION",
        "error_trace": "Terminal failure: all replicas exhausted. Incident paged."
    }

def route_after_primary(state: ResilientState) -> str:
    if state.get("data_payload"):
        return END
    if state.get("retry_count", 0) > 2:
        return "dead_letter_alert"
    return "fallback_recovery"

# Construct graph with recovery branch
builder = StateGraph(ResilientState)
builder.add_node("primary_fetch", primary_data_node)
builder.add_node("fallback_recovery", fallback_recovery_node)
builder.add_node("dead_letter_alert", dead_letter_alert_node)

builder.add_edge(START, "primary_fetch")
builder.add_conditional_edges(
    "primary_fetch",
    route_after_primary,
    {
        END: END,
        "fallback_recovery": "fallback_recovery",
        "dead_letter_alert": "dead_letter_alert"
    }
)
builder.add_edge("fallback_recovery", END)
builder.add_edge("dead_letter_alert", END)

app = builder.compile(checkpointer=MemorySaver())
print("Resilient Failure Handling Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Failure Injection and Circuit Breaker Simulator", divider="rainbow")

st.markdown("""
Test how different failure types (transient timeouts, tool exceptions, context limits) behave under configured resilience parameters.
""")

sim = FailureHandlingSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("🛠️ Simulation 1: Node Retry with Backoff & Jitter")
    st.markdown("Configure a node with an injected failure probability and observe retry intervals.")
    
    node_name_input = st.selectbox("Target Node", ["api_indexer", "database_writer", "vector_embedder"])
    fail_rate_slider = st.slider("Simulated Failure Probability", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    max_retries_slider = st.slider("Max Retries Allowed", min_value=1, max_value=5, value=3)
    base_delay_input = st.slider("Initial Delay (seconds)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
    enable_jitter = st.toggle("Enable Uniform Jitter", value=True)

    if st.button("🚀 Trigger Node Execution"):
        result = sim.execute_node_with_retry(
            node_name=node_name_input,
            fail_rate=fail_rate_slider,
            max_retries=max_retries_slider,
            base_delay_sec=base_delay_input,
            jitter=enable_jitter
        )
        if result["success"]:
            st.success(f" Node `{result['node']}` succeeded after {result['total_attempts']} attempt(s)!")
        else:
            st.error(f"❌ Node `{result['node']}` failed terminally after reaching max retries ({result['total_attempts']}).")
        
        df_log = pd.DataFrame(result["logs"])
        st.dataframe(df_log, use_container_width=True)

with col_sim2:
    st.subheader("🧰 Simulation 2: Tool Sandboxing & Model Cascade")
    st.markdown("Evaluate how `handle_tool_errors` protects the graph from unhandled tool crashes.")
    
    tool_select = st.selectbox("Simulate Tool", ["sql_query_tool", "web_scraper", "stripe_charge_tool"])
    crash_tool = st.toggle("Inject Tool Crash (e.g. 500 API Error)", value=True)
    sandbox_toggle = st.toggle("Enable handle_tool_errors Sandboxing", value=True)

    if st.button("⚡ Execute Tool Call"):
        t_res = sim.simulate_tool_execution(tool_select, sandboxed=sandbox_toggle, crash_simulation=crash_tool)
        if t_res["graph_crashed"]:
            st.error(f"💥 GRAPH CRASHED: {t_res['result']}")
        elif t_res["status"] == "HANDLED_ERROR":
            st.warning(f"🛡️ Tool Error Handled Gracefully: {t_res['result']}")
        else:
            st.success(f" {t_res['result']}")

    st.markdown("---")
    st.markdown("#### Model Cascade (Context Window Overflow)")
    token_count = st.number_input("Injected Prompt Tokens", min_value=1000, max_value=200000, value=95000, step=5000)
    
    if st.button("🔄 Evaluate Model Router Cascade"):
        c_res = sim.simulate_model_cascade(
            primary_model="gpt-4o-8k-fast",
            fallback_model="claude-3-5-sonnet-128k",
            primary_context_limit=32000,
            prompt_tokens=token_count
        )
        if c_res["used_fallback"]:
            st.warning(f"⚠️ {c_res['message']} (Active Model: `{c_res['active_model']}`)")
        else:
            st.success(f" {c_res['message']} (Active Model: `{c_res['active_model']}`)")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What happens to graph state when a node throws an unhandled exception?", expanded=False):
    st.markdown("""
    **Answer**:
    In LangGraph, all state mutations committed prior to the failing superstep remain preserved in the checkpointer.
    The writes of the failing superstep itself are aborted and rolled back.
    The graph thread pauses in an error state. You can inspect the last valid checkpoint with `app.get_state(config)`.
    """)

with st.expander("Q2: How does ToolNode handle tool execution errors without breaking the agent loop?", expanded=False):
    st.markdown("""
    **Answer**:
    When `handle_tool_errors=True` (or a custom callable error handler is passed), `ToolNode` wraps individual tool executions in a `try/except` block.
    Upon catching an exception, it generates a synthetic `ToolMessage` containing the error text and an error status flag.
    This message is added to the conversation history, allowing the LLM in the next superstep to read the error and correct its behavior.
    """)

with st.expander("Q3: When should you use RetryPolicy vs a conditional edge for error handling?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Use `RetryPolicy`** for **transient infrastructure glitches** (temporary socket disconnects, 429 rate limits, 503 service unavailable) where doing the exact same operation after a short backoff is likely to succeed.
    - **Use conditional edges** for **semantic or logic failures** (invalid user credentials, context length limits, business rule violations) where re-attempting identical code will fail again, requiring an alternate code path or fallback model.
    """)

with st.expander("Q4: Why must jitter always be applied to exponential backoff algorithms?", expanded=False):
    st.markdown("""
    **Answer**:
    Without jitter, if 100 concurrent agent threads fail simultaneously during a database hiccup, all 100 threads will wake up and retry at the exact same millisecond (e.g. at 1s, 2s, 4s).
    This creates the classic **Thundering Herd** problem, immediately knocking the database back down.
    Jitter introduces pseudo-random variation into the sleep duration, spreading retry spikes across a wider time interval.
    """)

with st.expander("Q5: How does LangGraph handle context window limits in multi-turn agent chats?", expanded=False):
    st.markdown("""
    **Answer**:
    Context limit overflow can be mitigated at three levels:
    1. **Pre-emptive trimming**: Using `trim_messages()` or sliding-window message reducers in state channels.
    2. **Model cascades**: Using `.with_fallbacks()` to seamlessly swap to a larger-window model if the primary model throws a context length error.
    3. **Summarization nodes**: A conditional branch that triggers an LLM summarization sub-graph when message token count crosses an SLA threshold.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Retry Loops on Semantic 400 Bad Request Errors")
    st.markdown("""
    *Mechanism*: Configuring a `RetryPolicy` on a node without filtering exception types.  
    *Consequence*: If an API returns HTTP 400 (e.g., malformed JSON payload), the node retries 5 times, wasting compute and delaying error propagation.  
    *Fix*: Always pass `retry_on=(ConnectionError, TimeoutError, RateLimitError)` and never retry on 400/404.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Tool Exceptions Crashing the Parent Process")
    st.markdown("""
    *Mechanism*: Instantiating `ToolNode(tools=...)` without enabling `handle_tool_errors=True`.  
    *Consequence*: A single unexpected 3rd-party API crash halts the entire user conversation.  
    *Fix*: Explicitly set `handle_tool_errors=True` or provide a custom error formatting function.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Hidden Memory Leaks in Long-Running Dead-Letter States")
    st.markdown("""
    *Mechanism*: Continuously accumulating failed thread IDs in an in-memory dictionary without a TTL or database persistence.  
    *Consequence*: Over days of continuous production load, dead-letter monitoring exhausts worker RAM.  
    *Fix*: Route dead-letter events to external log aggregation or message queues (Kafka/RabbitMQ).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Reliability Warning): Synchronous Blocking Calls in Async Nodes")
    st.markdown("""
    *Mechanism*: Calling synchronous `time.sleep()` or synchronous `requests.get()` inside async node functions (`async def`).  
    *Consequence*: Blocks the entire asyncio event loop, freezing all concurrent agent sessions on the container.  
    *Fix*: Use `asyncio.sleep()` and async HTTP clients like `httpx.AsyncClient` or `aiohttp`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**handle_tool_errors**\\nConverts tool Python exceptions into synthetic ToolMessages, enabling LLM self-correction.")
with col_fc2:
    st.info("**RetryPolicy**\\nConfigures exponential backoff, maximum attempts, and jitter for transient node failures.")
with col_fc3:
    st.info("**with_fallbacks**\\nLangChain abstraction for seamless failover between LLM providers and models.")
with col_fc4:
    st.info("**Dead-Letter Routing**\\nA conditional branch directing terminally failed states to human escalation or alerting.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the difference between node-level, tool-level, and model-level failures.", value=True)
c2 = st.checkbox("I know how handle_tool_errors transforms Python exceptions into conversational feedback.", value=True)
c3 = st.checkbox("I can configure a RetryPolicy with exponential backoff and jitter in LangGraph.", value=True)
c4 = st.checkbox("I know how to implement cross-provider model fallbacks using with_fallbacks.", value=True)
c5 = st.checkbox("I understand why transient errors require retries while semantic errors require routing.", value=True)
c6 = st.checkbox("I know how to inspect thread state after a timeout using app.get_state(config).", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 17.2 'Failure Handling' with full architectural and practical comprehension!")
