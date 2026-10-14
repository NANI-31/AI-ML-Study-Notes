import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="5.3 Node Engineering - LangGraph Curriculum",
    page_icon="🛠️",
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
    .badge-node {
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
# MODULE-SCOPE RESILIENCE & CHAOS SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class NodeResilienceSimulator:
    """
    Simulates enterprise node engineering patterns:
    idempotency tokens, retry policies with backoff, chaos fault injection,
    and side-effect isolation.
    """
    def __init__(self):
        self.processed_idempotency_keys: set = set()
        self.execution_history: List[Dict[str, Any]] = []

    def execute_resilient_node(self, payload: Dict[str, Any], idempotency_key: str, failure_rate: float, max_retries: int) -> Dict[str, Any]:
        """
        Executes a simulated node with idempotency checking and retry backoff.
        """
        # 1. Idempotency Guard
        if idempotency_key in self.processed_idempotency_keys:
            record = {
                "idempotency_key": idempotency_key,
                "status": "CACHE_HIT (Idempotent replay)",
                "attempts": 0,
                "output": "Duplicate request skipped; returned cached result.",
                "is_duplicate": True
            }
            self.execution_history.append(record)
            return record

        # 2. Execution with simulated retry loop
        attempts = 0
        success = False
        last_error = None

        while attempts <= max_retries:
            attempts += 1
            # Simulate transient failure
            rand_val = np.random.uniform(0.0, 1.0)
            if rand_val < failure_rate:
                last_error = f"Transient 503 Service Unavailable (Attempt {attempts})"
            else:
                success = True
                break

        if success:
            self.processed_idempotency_keys.add(idempotency_key)
            record = {
                "idempotency_key": idempotency_key,
                "status": "SUCCESS",
                "attempts": attempts,
                "output": f"Processed payload for {payload.get('item_id', 'item_unknown')}",
                "is_duplicate": False
            }
        else:
            record = {
                "idempotency_key": idempotency_key,
                "status": "FAILED (Max retries exceeded)",
                "attempts": attempts,
                "output": f"Error: {last_error}",
                "is_duplicate": False
            }

        self.execution_history.append(record)
        return record


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Node Engineering, Idempotency, and Failure Tolerance", divider="rainbow")

st.markdown("""
In mission-critical enterprise environments, nodes do not run in sterile test scenarios.
They encounter transient network timeouts, third-party LLM rate limits (HTTP 429), partial database write failures,
and concurrent graph replays. **Node Engineering** is the practice of designing nodes that are:
1. **Idempotent**: Re-executing a node with the same input yields identical state without duplicate side-effects.
2. **Pure & Isolated**: Free of uncontrolled side-effects to ensure deterministic replayability.
3. **Fault-Tolerant**: Equipped with exponential retry backoff and defensive fallback routing.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    resilience_blueprint_dot = """
    digraph NodeEngineeringBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress_guard {
            label="1. Node Ingress & Idempotency Filter";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="Node Invocation Request", fillcolor="#1E293B", shape=ellipse];
            Idempotency_Check [label="Idempotency Check\\nHas token been committed?", shape=diamond, fillcolor="#0284C7"];
            Skip_Duplicate [label="Return Cached State\\n(Bypasses Side Effects)", fillcolor="#059669"];
        }

        subgraph cluster_retry_circuit {
            label="2. Fault-Tolerant Execution Circuit";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Execute_Core [label="Execute Primary Node Logic\\n(LLM / External API / DB)", fillcolor="#334155"];
            Evaluate_Result [label="Success or Transient Error?", shape=diamond, fillcolor="#4338CA"];
            Backoff_Timer [label="Exponential Backoff\\nDelay = base * 2^attempt + jitter", fillcolor="#6D28D9"];
            Fallback_Handler [label="Fallback Degradation\\nEmit graceful error delta", fillcolor="#DC2626"];
        }

        subgraph cluster_egress_state {
            label="3. Verified Egress & State Commit";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            Commit_Token [label="Record Idempotency Token in State", fillcolor="#059669"];
            Emit_Delta [label="Emit Partial State Delta", fillcolor="#059669"];
            END [label="Proceed to Next Superstep", fillcolor="#1E293B", shape=ellipse];
        }

        START -> Idempotency_Check;
        Idempotency_Check -> Skip_Duplicate [label="Yes: Token exists"];
        Idempotency_Check -> Execute_Core [label="No: New turn"];
        Execute_Core -> Evaluate_Result;
        Evaluate_Result -> Commit_Token [label="Success (200 OK)"];
        Evaluate_Result -> Backoff_Timer [label="Transient Error (attempts < max)"];
        Backoff_Timer -> Execute_Core [label="Retry attempt"];
        Evaluate_Result -> Fallback_Handler [label="Retries Exhausted"];
        Fallback_Handler -> Emit_Delta [label="Graceful error delta"];
        Commit_Token -> Emit_Delta;
        Emit_Delta -> END;
        Skip_Duplicate -> END;
    }
    """
    st.graphviz_chart(resilience_blueprint_dot, use_container_width=True)

st.markdown("""
### The Three Invariants of Resilient Node Engineering
- **Zero In-Place Mutation**: Input state must remain immutable to protect checkpoint rollbacks.
- **Side-Effect Quarantine**: All external I/O (Stripe charges, email dispatch) must be guarded by idempotency tokens.
- **Fail-Forward Design**: Nodes catch expected domain errors and emit structured error states rather than crashing the Python process.
""")

# ==============================================================================
# SECTION 2: STATELESS NODES
# ==============================================================================
st.header("2. Stateless Nodes: Decoupled Reusable Compute Units", divider="rainbow")

st.markdown("""
A **Stateless Node** operates purely on standard inputs and returns outputs without depending on or assuming
the existence of external mutable state or previous invocation history.
""")

col_sl1, col_sl2 = st.columns(2)
with col_sl1:
    st.markdown("""
    #### Characteristics of Stateless Nodes
    - **Portability**: The same node function can be registered across multiple different graphs with different state schemas.
    - **Trivial Unit Testing**: You can test the function by simply passing a test dictionary without spinning up a LangGraph runtime.
    - **Zero Memory Leaks**: Does not cache state in global variables or instance attributes.
    """)
with col_sl2:
    code_stateless = """def sanitize_input_node(state: dict) -> dict:
    # Pure stateless transformation
    raw = state.get("user_query", "")
    return {"cleaned_query": raw.strip()}
"""
    st.code(code_stateless, language="python")

# ==============================================================================
# SECTION 3: STATEFUL NODES
# ==============================================================================
st.header("3. Stateful Nodes: Schema-Coupled State Mutations and Channel Reducers", divider="rainbow")

st.markdown("""
A **Stateful Node** is deeply coupled to the specific state schema of a graph.
It expects specific channels to exist, reads historical context, increments counters, and triggers channel reducers.
""")

code_stateful = """from typing import TypedDict, Annotated, List
import operator

class MultiTurnState(TypedDict):
    turn_count: int
    chat_history: Annotated[List[str], operator.add]
    is_active: bool

def conversation_turn_node(state: MultiTurnState) -> dict:
    # Strongly coupled to MultiTurnState channels
    current_turn = state["turn_count"]
    return {
        "turn_count": current_turn + 1,
        "chat_history": [f"Turn {current_turn + 1} completed"],
        "is_active": current_turn < 10
    }
"""
st.code(code_stateful, language="python")

# ==============================================================================
# SECTION 4: PURE FUNCTIONS
# ==============================================================================
st.header("4. Pure Functions: Determinism, Concurrency Safety, and Replayability", divider="rainbow")

st.markdown("""
In functional programming, a function is **pure** if:
1. Given the exact same inputs, it always produces the exact same output.
2. It causes zero observable side-effects (no disk writes, no network calls, no modifying external globals).
""")

st.markdown("""
#### Why LangGraph Favors Pure Nodes for Routing and Data Processing:
- **Time-Travel Debugging**: In LangGraph Studio, you can rewind state to superstep 3 and replay forward. If nodes are pure, replay produces 100% identical states.
- **Parallel Branching**: Two pure nodes running in parallel will never suffer race conditions or cross-thread interference.
""")

# ==============================================================================
# SECTION 5: SIDE EFFECTS
# ==============================================================================
st.header("5. Side Effects: External I/O Boundaries, Auditing, and Transactional Integrity", divider="rainbow")

st.markdown("""
AI agents ultimately exist to take real-world actions: writing to databases, charging credit cards, or sending emails.
These operations are **side-effects**. Because LangGraph graphs can retry, pause, and fork, side-effects must be carefully isolated.
""")

col_se1, col_se2 = st.columns(2)
with col_se1:
    st.error("❌ Uncontrolled Side-Effect")
    st.code("""def bad_email_node(state: dict) -> dict:
    # DANGEROUS: If this graph pauses or retries,
    # the customer gets 5 duplicate emails!
    send_smtp_email(to=state["email"], body="Confirmed")
    return {"status": "SENT"}
""", language="python")

with col_se2:
    st.success("✅ Idempotent Guarded Side-Effect")
    st.code("""def safe_email_node(state: dict) -> dict:
    tx_id = state["transaction_id"]
    if tx_id in state.get("completed_tx_ids", []):
        return {} # Skip duplicate execution!
        
    send_smtp_email(to=state["email"], idempotency_key=tx_id)
    return {"completed_tx_ids": [tx_id], "status": "SENT"}
""", language="python")

# ==============================================================================
# SECTION 6: ERROR HANDLING
# ==============================================================================
st.header("6. Error Handling: Graceful Degradation, Fallback Paths, and Exception Bubbling", divider="rainbow")

st.markdown("""
A production node should never allow unexpected runtime exceptions to bubble unhandled and terminate the server process.
Instead, adopt the **Fail-Forward Pattern**: capture the exception and update state with diagnostic error information.
""")

code_fail_forward = """def resilient_llm_node(state: dict) -> dict:
    try:
        response = call_primary_model(state["prompt"])
        return {"response": response, "error": None}
    except Exception as exc:
        # Fail forward: state records failure so graph router can handle it
        return {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "failed_at_step": "primary_llm"
            },
            "should_fallback": True
        }
"""
st.code(code_fail_forward, language="python")

# ==============================================================================
# SECTION 7: RETRYABLE NODES
# ==============================================================================
st.header("7. Retryable Nodes: Exponential Backoff, Jitter, and Transient Fault Recovery", divider="rainbow")

st.markdown("""
LangGraph supports native node retry policies during graph compilation via `RetryPolicy`:
""")

code_retry_policy = """from langgraph.graph import StateGraph
from langgraph.prebuilt import RetryPolicy

# Configure automated exponential backoff
custom_retry_policy = RetryPolicy(
    initial_interval=1.0,  # Wait 1s after first failure
    backoff_factor=2.0,    # Double wait time each attempt: 1s, 2s, 4s...
    max_interval=30.0,     # Cap maximum backoff delay
    max_attempts=3,        # Up to 3 retries before raising exception
    retry_on=(ConnectionError, TimeoutError) # Specific retryable errors
)

builder = StateGraph(MyState)
# Bind retry policy directly to node registration!
builder.add_node("flaky_api_node", flaky_api_call, retry=custom_retry_policy)
"""
st.code(code_retry_policy, language="python")

# ==============================================================================
# SECTION 8: IDEMPOTENT NODES
# ==============================================================================
st.header("8. Idempotent Nodes: De-duplication Keys and Safe Re-Execution Semantics", divider="rainbow")

st.markdown("""
**Idempotency** guarantees that executing an operation multiple times with identical parameters
yields the same end result as executing it once.
""")

st.markdown("""
#### Implementing Idempotency in LangGraph Nodes:
1. **Derive Deterministic Idempotency Key**: Combine `thread_id` + `node_name` + `step_index` or payload hash.
2. **Consult State Registry**: Check if the idempotency key already exists in a dedicated `processed_keys: set` channel.
3. **Short-Circuit on Cache Hit**: If present, return `{}` or cached values immediately without performing external calls.
4. **Commit Token**: If executed, append the key to the state registry.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Resilient Production Node with Retry and Idempotency", divider="rainbow")

st.markdown("""
Below is a production-grade implementation of a resilient financial ledger update node,
incorporating idempotency checking, exponential backoff simulation, and graceful error routing.
""")

prod_resilient_code = '''import time
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: Define Resilient State
class LedgerState(TypedDict):
    transaction_id: str
    amount_cents: int
    account_id: str
    committed_tx_keys: Annotated[List[str], operator.add]
    audit_logs: Annotated[List[str], operator.add]
    status: str
    error_details: Optional[Dict[str, Any]]

# Step 2: Implement Resilient Idempotent Node
def process_ledger_transaction_node(state: LedgerState) -> Dict[str, Any]:
    tx_id = state["transaction_id"]
    committed_keys = state.get("committed_tx_keys", [])
    
    # 1. Idempotency Verification
    if tx_id in committed_keys:
        return {
            "status": "DUPLICATE_IGNORED",
            "audit_logs": [f"Transaction {tx_id} already committed. Replay ignored."]
        }
    
    # 2. Resilient Execution with Local Retries
    max_retries = 3
    base_delay = 0.1
    
    for attempt in range(1, max_retries + 1):
        try:
            # Simulate external payment gateway / ledger write
            if state["amount_cents"] <= 0:
                raise ValueError("Transaction amount must be positive.")
                
            # Simulated successful commit
            return {
                "committed_tx_keys": [tx_id],
                "status": "COMMITTED",
                "audit_logs": [f"Successfully processed tx {tx_id}: ${state['amount_cents']/100:.2f}"],
                "error_details": None
            }
        except ValueError as val_err:
            # Non-retryable business validation error
            return {
                "status": "REJECTED",
                "audit_logs": [f"Validation failure for tx {tx_id}: {val_err}"],
                "error_details": {"type": "ValidationError", "msg": str(val_err)}
            }
        except Exception as trans_err:
            # Transient error retry
            if attempt == max_retries:
                return {
                    "status": "FAILED",
                    "audit_logs": [f"Max retries exhausted for tx {tx_id}: {trans_err}"],
                    "error_details": {"type": "TransientError", "msg": str(trans_err)}
                }
            time.sleep(base_delay * (2 ** (attempt - 1)))
            
    return {"status": "UNKNOWN_ERROR"}

# Build Graph
def create_ledger_graph():
    builder = StateGraph(LedgerState)
    builder.add_node("process_tx", process_ledger_transaction_node)
    builder.add_edge(START, "process_tx")
    builder.add_edge("process_tx", END)
    return builder.compile()

if __name__ == "__main__":
    app = create_ledger_graph()
    sample_tx = {
        "transaction_id": "tx_2026_9011",
        "amount_cents": 4500,
        "account_id": "acc_enterprise_99",
        "committed_tx_keys": [],
        "audit_logs": ["Ledger pipeline started."],
        "status": "PENDING",
        "error_details": None
    }
    # First invocation: commits
    out1 = app.invoke(sample_tx)
    print("Run 1 Status:", out1["status"])
    
    # Second invocation (duplicate replay): idempotent short-circuit
    out2 = app.invoke(out1)
    print("Run 2 Status:", out2["status"])
'''
st.code(prod_resilient_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Chaos Engineering Simulator and Retry Tester", divider="rainbow")

st.markdown("""
Test node resilience under chaos conditions. Inject transient failure probabilities,
configure retry policies, and test idempotency protections against duplicate invocations.
""")

resilience_sim = NodeResilienceSimulator()

col_ch_l, col_ch_r = st.columns([1, 1.2])

with col_ch_l:
    st.subheader("Chaos Engine Configuration")
    test_tx_key = st.text_input("Idempotency Key", value="IDEMP_KEY_ALPHA_001")
    item_desc = st.text_input("Item ID / Payload", value="SKU_GPU_H100_CLUSTER")
    
    chaos_failure_rate = st.slider("Simulated Failure Rate (Chaos Factor)", 0.0, 1.0, 0.40)
    max_retry_attempts = st.number_input("Max Retry Attempts", min_value=1, max_value=5, value=3)

    if st.button("🔥 Execute Node with Chaos Injection", use_container_width=True):
        payload = {"item_id": item_desc}
        res = resilience_sim.execute_resilient_node(
            payload,
            test_tx_key,
            chaos_failure_rate,
            int(max_retry_attempts)
        )
        st.session_state.last_chaos_run = res
        if res["status"] == "SUCCESS":
            st.success(f"Execution Succeeded on attempt {res['attempts']}!")
        elif "CACHE_HIT" in res["status"]:
            st.info("Idempotent Cache Hit: Duplicate replay skipped.")
        else:
            st.error(f"Execution Failed: {res['output']}")

with col_ch_r:
    st.subheader("Execution Telemetry & Audit Trail")
    if "last_chaos_run" in st.session_state:
        st.json(st.session_state.last_chaos_run)
    else:
        st.info("Execute a node to generate telemetry records.")

    st.markdown("**Processed Idempotency Key Cache:**")
    st.write(list(resilience_sim.processed_idempotency_keys))

st.markdown("### Chaos Run History")
if resilience_sim.execution_history:
    st.dataframe(pd.DataFrame(resilience_sim.execution_history), use_container_width=True)
else:
    st.caption("No executions recorded in this session.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does RetryPolicy in LangGraph handle transient errors vs unrecoverable errors?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph's `RetryPolicy(retry_on=(...))` allows specifying an explicit tuple of exception classes (e.g. `(httpx.RequestError, openai.RateLimitError)`).
    If a raised exception is an instance of `retry_on`, LangGraph applies exponential backoff and retries the node up to `max_attempts`.
    If the exception is unrecoverable (such as `ValueError`, `KeyError`, or `PydanticCustomError`), LangGraph immediately bubbles the exception without wasting time or quota on futile retries.
    """)

with st.expander("Q2: Why is idempotency critical when combining LangGraph with human-in-the-loop interrupts?", expanded=False):
    st.markdown("""
    **Answer**:
    When a human approval node pauses execution, the superstep state is checkpointed.
    If the graph is resumed, or if a user forks and re-runs the previous node, nodes preceding the interrupt might be evaluated again. Without idempotency guards, external actions like sending notification emails or reserving inventory would execute a second time.
    """)

with st.expander("Q3: What are the risks of using global variables for state caching inside a node function?", expanded=False):
    st.markdown("""
    **Answer**:
    Storing state in Python module globals (`GLOBAL_CACHE[session_id] = ...`):
    1. Breaks horizontal scalability when running LangGraph across multiple worker nodes or Kubernetes pods.
    2. Causes race conditions in async multi-tenant servers where multiple concurrent requests access the same global memory.
    3. Defeats LangGraph's checkpointer, as global variables are not persisted in SQLite or Postgres checkpoints.
    """)

with st.expander("Q4: What is the Fail-Forward pattern and why is it preferred over raw exception bubbling?", expanded=False):
    st.markdown("""
    **Answer**:
    Raw exception bubbling crashes the graph execution thread, terminating the user's conversation abruptly.
    In the **Fail-Forward pattern**, the node catches the error, packages the diagnostic context into a state channel (e.g. `{"has_error": True, "error_msg": str(e)}`), and returns cleanly. Downstream conditional edges can then route the graph to a fallback LLM, a human operator, or a user-friendly recovery prompt.
    """)

with st.expander("Q5: How does adding jitter to exponential backoff prevent the Thundering Herd problem?", expanded=False):
    st.markdown("""
    **Answer**:
    Without jitter, if an external API experiences a brief outage and 1,000 parallel agent nodes fail simultaneously, all 1,000 nodes will calculate the exact same backoff delay (e.g. 2.0s) and retry at the exact same millisecond, crashing the recovering API again. Adding random jitter (`delay = base * 2^attempt + random_uniform(0, 0.5)`) spreads retry requests smoothly across time.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Non-Idempotent External Actions on Graph Retries")
    st.markdown("""
    *Mechanism*: Charging credit cards or sending notifications inside a node that has an automatic `RetryPolicy` without an idempotency key.  
    *Consequence*: If the node succeeds in charging the card but times out reading the response, the retry charges the card a second time!  
    *Fix*: Always pass unique idempotency keys to external payment and message gateways.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Catching BaseException and Masking Cancellation Signals")
    st.markdown("""
    *Mechanism*: Writing `except BaseException:` inside a node function.  
    *Consequence*: Catches `asyncio.CancelledError` and `KeyboardInterrupt`, preventing LangGraph from cleanly shutting down threads or handling timeouts.  
    *Fix*: Strictly catch `except Exception:` or specific domain exception classes.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Infinite Retry Storms from Broad Retry Catching")
    st.markdown("""
    *Mechanism*: Configuring `RetryPolicy(retry_on=Exception)` with high attempt counts without checking error status codes.  
    *Consequence*: Retries 401 Unauthorized or 400 Bad Request errors repeatedly, wasting seconds of latency and quota on unrecoverable bugs.  
    *Fix*: Restrict `retry_on` to transient network and rate-limit exceptions (e.g. 429, 503).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Hidden State Mutation via In-Memory Global Caches")
    st.markdown("""
    *Mechanism*: Mutating a process-local in-memory dict as a pseudo-database inside a node.  
    *Consequence*: In serverless deployments (AWS Lambda, Cloud Run), instances terminate unpredictably, resulting in silent data loss.  
    *Fix*: Always persist state inside LangGraph channels or external centralized databases (Redis, Postgres).
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Stateless Node**\\nA pure compute function that depends solely on its explicit inputs without external state dependencies.")
with col_fc2:
    st.info("**RetryPolicy**\\nA LangGraph compiler configuration defining automated exponential backoff parameters for a node.")
with col_fc3:
    st.info("**Idempotency**\\nThe property where repeated execution with identical parameters produces the exact same state without duplicate side effects.")
with col_fc4:
    st.info("**Fail-Forward**\\nAn error pattern where exceptions are packaged into state channels to enable graceful graph recovery.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know when to design a node as stateless versus stateful.", value=True)
c2 = st.checkbox("I understand the importance of pure functions for deterministic time-travel debugging.", value=True)
c3 = st.checkbox("I can guard external side-effects using unique idempotency keys.", value=True)
c4 = st.checkbox("I know how to implement the Fail-Forward pattern for graceful error handling.", value=True)
c5 = st.checkbox("I can configure native LangGraph RetryPolicy with exponential backoff and jitter.", value=True)
c6 = st.checkbox("I understand how to prevent infinite retry storms by targeting specific transient exception types.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 5.3 'Node Engineering' with full architectural and practical comprehension!")
