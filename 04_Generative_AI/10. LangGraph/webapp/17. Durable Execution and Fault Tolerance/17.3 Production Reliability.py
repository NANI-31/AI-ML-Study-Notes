import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="17.3 Production Reliability - LangGraph Curriculum",
    page_icon="⚙️",
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
    .badge-ok {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-warn {
        background-color: #382B14;
        color: #FBBF24;
        border: 1px solid #D97706;
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
    Simulates production transaction semantics and recovery mechanics:
    - Idempotency key tracking
    - Pending write staging buffer inspection
    - Parallel superstep partial-failure recovery
    - At-least-once vs exactly-once semantics verification
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.processed_idempotency_keys = set()
        self.database_records = []
        self.pending_writes_buffer = []
        self.step_counter = 0

    def execute_transaction(
        self,
        transaction_id: str,
        amount_usd: float,
        enforce_idempotency: bool,
        crash_before_commit: bool
    ) -> Dict[str, Any]:
        self.step_counter += 1
        
        # Check idempotency
        if enforce_idempotency and transaction_id in self.processed_idempotency_keys:
            return {
                "step": self.step_counter,
                "tx_id": transaction_id,
                "status": "DEDUPLICATED",
                "message": f"Idempotency token '{transaction_id}' already committed. Replaying stored outcome without duplicate side-effect.",
                "total_charged": sum(r["amount"] for r in self.database_records),
                "committed": True
            }

        # Stage in pending writes
        write_entry = {"tx_id": transaction_id, "amount": amount_usd, "step": self.step_counter}
        self.pending_writes_buffer.append(write_entry)

        if crash_before_commit:
            # Crash happens after staging, before checkpoint commit
            # In Pregel, uncommitted pending writes are discarded on node crash
            self.pending_writes_buffer.clear()
            return {
                "step": self.step_counter,
                "tx_id": transaction_id,
                "status": "CRASHED_BEFORE_COMMIT",
                "message": f"Worker crashed during superstep! Pending write for '{transaction_id}' discarded. No database mutation committed.",
                "total_charged": sum(r["amount"] for r in self.database_records),
                "committed": False
            }

        # Commit phase
        self.database_records.append(write_entry)
        if enforce_idempotency:
            self.processed_idempotency_keys.add(transaction_id)
        self.pending_writes_buffer.clear()

        return {
            "step": self.step_counter,
            "tx_id": transaction_id,
            "status": "COMMITTED",
            "message": f"Transaction '{transaction_id}' for {amount_usd} USD committed to checkpointer and ledger.",
            "total_charged": sum(r["amount"] for r in self.database_records),
            "committed": True
        }

    def simulate_parallel_superstep(
        self,
        nodes: List[str],
        failing_node: Optional[str]
    ) -> Dict[str, Any]:
        results = {}
        for n in nodes:
            if n == failing_node:
                results[n] = {"status": "FAILED", "committed": False}
            else:
                results[n] = {"status": "COMPLETED", "committed": False}  # Not committed because superstep atomic barrier failed!
        
        all_ok = failing_node is None
        return {
            "all_succeeded": all_ok,
            "node_statuses": results,
            "superstep_committed": all_ok,
            "explanation": "If any parallel node fails in a superstep, the entire superstep barrier aborts and zero writes are committed." if not all_ok else "All parallel nodes completed successfully; atomic barrier committed state."
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production Reliability and Transactional Guarantees", divider="rainbow")

st.markdown("""
Deploying LangGraph agents in production enterprise environments requires rock-solid reliability guarantees.
Unlike toy demos that can simply crash and restart from scratch, production systems must handle:
- **Financial and Database Idempotency**: Re-running a node due to network retries must never double-bill a customer or duplicate records.
- **Side-Effect Isolation**: Keeping pure state transitions decoupled from external network side-effects.
- **Pending Writes Buffering**: How the Pregel execution engine stages partial node outputs before committing an atomic superstep barrier.
- **Distributed Delivery Guarantees**: Understanding where LangGraph provides **at-least-once** execution and how application designers achieve **effectively-once** transaction semantics.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ReliabilityArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_orchestrator {
            label="1. Superstep Barrier & Worker Node Execution";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            worker_a [label="Worker Node A (Compute)\\nPure State Transformation", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            worker_b [label="Worker Node B (External Side-Effect)\\nIdempotency Token Enforced", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_staging {
            label="2. Pregel Staging Memory (Pending Writes)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            staging_buffer [label="Pending Writes Buffer\\ncheckpoint_writes Staging Queue", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            atomic_barrier [label="Atomic Superstep Barrier\\nAll Nodes Evaluated?", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_persistence {
            label="3. Durable Storage & External Ledger";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            postgres_checkpointer [label="Relational Checkpointer (PostgresSaver)\\nCommitted Checkpoint ID", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            external_ledger [label="External Systems / Payment Gateway\\nDeduplication Ledger via Token", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            discard_bin [label="Rollback / Discard Buffer\\nZero State Pollution", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
        }

        worker_a -> staging_buffer [label="Yields Channel Delta"];
        worker_b -> staging_buffer [label="Yields External Output"];
        staging_buffer -> atomic_barrier;

        atomic_barrier -> postgres_checkpointer [label="Commit: All Nodes Succeeded", color="#34D399"];
        atomic_barrier -> external_ledger [label="Idempotent External Sync", color="#34D399"];
        atomic_barrier -> discard_bin [label="Abort: Any Node Crashed", color="#EF4444", style=dashed];
        discard_bin -> worker_a [label="Re-execute Superstep", color="#FBBF24", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: IDEMPOTENCY
# ==============================================================================
st.header("2. Idempotency: Designing Deterministic Nodes and Deduplication Tokens", divider="rainbow")

st.markdown("""
An operation is **idempotent** if applying it multiple times produces the exact same outcome as applying it once:
`f(f(x)) = f(x)`

In an agentic graph with automated retries and recovery:
- A node might execute halfway, write to an external database, and then crash before returning its update dictionary to LangGraph.
- When the graph re-executes the crashed superstep, the node will be called **a second time**.
- If the node is not idempotent, duplicate rows, duplicate payments, or repeated notification emails will occur.
""")

col_id1, col_id2 = st.columns(2)
with col_id1:
    st.markdown("#### Anti-Pattern: Non-Idempotent Node")
    st.code('''# DANGEROUS: Increments counter or issues payment blindly
def charge_customer_node(state: OrderState) -> dict:
    # If the network breaks right after charge, retrying will charge twice!
    charge_resp = stripe.Charge.create(
        amount=state["total_cents"],
        currency="usd",
        customer=state["customer_id"]
    )
    return {"charge_id": charge_resp.id, "status": "PAID"}''', language="python")

with col_id2:
    st.markdown("#### Production Pattern: Idempotency Key Injection")
    st.code('''# PRODUCTION-GRADE: Deterministic Idempotency Key
def idempotent_charge_node(state: OrderState) -> dict:
    # Generate deterministic token from thread_id + step_id
    idempotency_token = f"order_{state['order_id']}_step_{state['payment_step']}"
    
    charge_resp = stripe.Charge.create(
        amount=state["total_cents"],
        currency="usd",
        customer=state["customer_id"],
        idempotency_key=idempotency_token # Stripe deduplicates duplicate calls!
    )
    return {"charge_id": charge_resp.id, "status": "PAID"}''', language="python")

# ==============================================================================
# SECTION 3: SIDE-EFFECT MANAGEMENT
# ==============================================================================
st.header("3. Side-Effect Management: Isolating State Mutations from External Network Calls", divider="rainbow")

st.markdown("""
To maintain clean recovery boundaries, production graphs strictly separate **pure computation nodes** from **external side-effect nodes**.
""")

st.markdown("""
1. **Pure Nodes (Deterministic State Transformations)**:
   - Calculate summaries, format strings, parse JSON, or route edges.
   - Depend *only* on the input state and return state dicts without network or filesystem calls.
   - Can be re-run an infinite number of times without risk.

2. **Side-Effect Nodes (External I/O Boundary)**:
   - Interact with APIs, send emails, write to Kafka, or charge credit cards.
   - Must be isolated into dedicated standalone nodes.
   - Must record their external response ID into state immediately.
""")

st.info("💡 **Best Practice Rule**: Never mix heavy LLM reasoning and irreversible external database mutations inside the same node function. Split them into a reasoning node followed by an action node.")

# ==============================================================================
# SECTION 4: RETRY POLICIES
# ==============================================================================
st.header("4. Retry Policies: Configuring Node-Level and Graph-Level Retries", divider="rainbow")

st.markdown("""
LangGraph provides fine-grained control over retries through `RetryPolicy`.
You can configure retry rules per-node during `builder.add_node()`.
""")

st.code('''from langgraph.types import RetryPolicy
import httpx

api_retry_policy = RetryPolicy(
    initial_interval=0.5,           # 500ms initial wait
    backoff_factor=2.0,             # Doubling interval (0.5s -> 1.0s -> 2.0s)
    max_interval=10.0,              # Max wait capped at 10s
    max_attempts=3,                 # Try up to 3 total attempts
    jitter=True,                    # Prevent synchronized spikes
    retry_on=(httpx.ConnectTimeout, httpx.ReadTimeout) # Only network timeouts
)

# builder.add_node("crm_sync_node", crm_sync_function, retry_policy=api_retry_policy)''', language="python")

st.markdown("""
**Key Retry Policy Attributes**:
- `initial_interval`: Duration (in seconds) to wait before the first retry attempt.
- `backoff_factor`: Multiplier applied to the interval after each failed attempt.
- `max_interval`: Upper ceiling limit on the sleep duration.
- `jitter`: Randomizes backoff intervals to prevent thundering herds across distributed workers.
- `retry_on`: Tuple of Python exception classes that trigger retries.
""")

# ==============================================================================
# SECTION 5: FAILURE RECOVERY
# ==============================================================================
st.header("5. Failure Recovery: Automated Worker Failover and State Reconstruction", divider="rainbow")

st.markdown("""
When a compute host running a LangGraph agent suffers a fatal crash (e.g. Kubernetes node termination, OOM kill, hardware failure), recovery proceeds through automated failover:
""")

st.markdown("""
1. **Worker Terminates**: The container hosting the thread dies abruptly mid-execution.
2. **State Remains Intact**: Because a database checkpointer (such as PostgreSQL) is used, all completed supersteps are already saved on the database host.
3. **New Worker Reclaims Thread**: The replacement worker or incoming request loads the latest checkpoint using the thread identifier:
   `snapshot = app.get_state(config={"configurable": {"thread_id": "session_99"}})`
4. **Resumption**: Calling `app.invoke(None, config=config)` instructs LangGraph to inspect `snapshot.next` and resume execution of the exact pending nodes that never finished committing.
""")

# ==============================================================================
# SECTION 6: PARTIAL EXECUTION RECOVERY
# ==============================================================================
st.header("6. Partial Execution Recovery: Resuming Half-Finished Parallel Supersteps", divider="rainbow")

st.markdown("""
Consider a superstep where two independent nodes execute concurrently in parallel:
- **Node A** (Fetch Stock Quote): Takes 100ms and finishes successfully.
- **Node B** (Query Financial News): Takes 800ms and crashes due to a timeout.

**What happens to Node A's output?**
In LangGraph's Pregel engine:
- The superstep operates as an **atomic barrier**.
- Node A's output is held in the internal staging buffer.
- Because Node B failed, the superstep barrier **does not commit**.
- When the graph is resumed, LangGraph will re-execute the superstep. Both nodes will execute from the beginning of that superstep unless specialized checkpointing semantics or separate subgraphs are employed.
""")

# ==============================================================================
# SECTION 7: PENDING WRITES
# ==============================================================================
st.header("7. Pending Writes: Managing checkpoint_writes Staging Buffers", divider="rainbow")

st.markdown("""
In LangGraph, state updates are not written to the checkpointer immediately as lines of code execute.
Instead, they are collected into a staging structure named `checkpoint_writes`.
""")

st.markdown("""
- Each node in a superstep produces a tuple `(node_name, channel_name, value)`.
- LangGraph stages these tuples in memory during the superstep.
- If all nodes in the superstep complete without unhandled exceptions, the runtime passes `checkpoint_writes` to the checkpointer adapter (e.g., `PostgresSaver`).
- The checkpointer commits the writes and moves the checkpoint pointer forward atomically.
- You can inspect pending writes during interrupt/human-in-the-loop states via:
  `snapshot.tasks[0].writes`
""")

st.code('''# Inspecting staged writes in a paused or interrupted graph
snapshot = app.get_state(config)
for task in snapshot.tasks:
    print(f"Task Name: {task.name}")
    print(f"Staged Writes: {task.writes}")''', language="python")

# ==============================================================================
# SECTION 8: EXACTLY-ONCE VS AT-LEAST-ONCE CONSIDERATIONS
# ==============================================================================
st.header("8. Exactly-Once vs At-Least-Once Considerations in Distributed State Machines", divider="rainbow")

st.markdown("""
In distributed systems theory, achieving genuine end-to-end **exactly-once execution** in the presence of arbitrary node crashes and network failures is physically impossible (the Two Generals' Problem).
Distributed engines offer **at-least-once execution**.
""")

col_eo1, col_eo2 = st.columns(2)
with col_eo1:
    st.markdown("#### The Guarantee Matrix")
    df_guarantees = pd.DataFrame({
        "Layer": ["LangGraph State Transitions", "External HTTP API Calls", "Database Insertions", "Agent LLM Invocations"],
        "Native Guarantee": ["Atomic per superstep", "At-least-once", "At-least-once", "At-least-once"],
        "Mechanism for Effectively-Once": ["Pregel barrier rollback", "Idempotency-Key header", "Upsert / Primary Key constraint", "Deterministic temperature & seed"]
    })
    st.dataframe(df_guarantees, hide_index=True, use_container_width=True)

with col_eo2:
    st.markdown("#### How to Achieve 'Effectively-Once' Semantics")
    st.markdown("""
    - **Use Unique Deduplication Keys**: Include `(thread_id, checkpoint_id, node_name)` in API request headers.
    - **Database Upserts**: Use SQL `ON CONFLICT (id) DO UPDATE` instead of raw `INSERT`.
    - **Compensating Transactions**: If a downstream step fails permanently, define a compensating cleanup node to rollback prior external changes.
    """)

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Fully Idempotent Transactional Agent Pipeline", divider="rainbow")

st.markdown("""
Here is a comprehensive production pattern demonstrating idempotent state management, deduplication token generation, and resilient persistence.
""")

st.code('''from typing import TypedDict, Annotated, Optional, List
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class TransactionState(TypedDict):
    order_id: str
    user_id: str
    amount_usd: float
    idempotency_token: str
    payment_status: str
    confirmation_code: Optional[str]
    audit_trail: List[str]

def prepare_transaction_node(state: TransactionState) -> dict:
    # Generate deterministic deduplication token for this order attempt
    token = f"tx_{state['order_id']}_{state['user_id']}"
    return {
        "idempotency_token": token,
        "payment_status": "PENDING",
        "audit_trail": state.get("audit_trail", []) + [f"Generated token {token}"]
    }

def idempotent_payment_gateway_node(state: TransactionState) -> dict:
    token = state["idempotency_token"]
    # Simulated external payment API call with idempotency key
    # In real systems, passing token guarantees duplicate requests do not charge again
    confirmation = f"CONF-{uuid.uuid5(uuid.NAMESPACE_DNS, token).hex[:8].upper()}"
    return {
        "payment_status": "CHARGED",
        "confirmation_code": confirmation,
        "audit_trail": state.get("audit_trail", []) + [f"Charged with token {token}"]
    }

def record_ledger_node(state: TransactionState) -> dict:
    return {
        "audit_trail": state.get("audit_trail", []) + [f"Ledger entry recorded: {state['confirmation_code']}"]
    }

# Build graph
builder = StateGraph(TransactionState)
builder.add_node("prepare", prepare_transaction_node)
builder.add_node("charge", idempotent_payment_gateway_node)
builder.add_node("record", record_ledger_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "charge")
builder.add_edge("charge", "record")
builder.add_edge("record", END)

app = builder.compile(checkpointer=MemorySaver())
print("Transactional Graph Compiled with Guaranteed Idempotency.")''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Idempotency & Pending Writes Recovery Laboratory", divider="rainbow")

st.markdown("""
Experiment with transactional execution, worker crashes, pending writes staging, and deduplication tokens.
""")

sim = ProductionReliabilitySimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("💳 Simulation 1: Transaction Idempotency & Worker Crashes")
    
    tx_id_input = st.text_input("Transaction ID (Idempotency Key)", value="TX-9042")
    charge_amount = st.number_input("Amount (USD)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
    enforce_idem = st.toggle("Enforce Idempotency Key Validation", value=True)
    crash_toggle = st.toggle("Simulate Worker Crash Before Checkpoint Commit", value=False)

    if st.button("⚡ Dispatch Transaction Node"):
        res = sim.execute_transaction(
            transaction_id=tx_id_input,
            amount_usd=charge_amount,
            enforce_idempotency=enforce_idem,
            crash_before_commit=crash_toggle
        )
        if res["status"] == "COMMITTED":
            st.success(f" {res['message']}")
        elif res["status"] == "DEDUPLICATED":
            st.info(f"🛡️ {res['message']}")
        else:
            st.error(f"💥 {res['message']}")

        st.metric("Total USD Charged in Ledger", f"${res['total_charged']:.2f}")

with col_wb2:
    st.subheader("⚡ Simulation 2: Parallel Superstep Atomic Barrier")
    st.markdown("Test what happens when one parallel node in a superstep crashes.")
    
    failing_node_select = st.selectbox(
        "Select Node to Crash (or None)",
        ["None (All Succeeded)", "node_query_vector_db", "node_fetch_user_profile"]
    )
    
    selected_fail = None if failing_node_select.startswith("None") else failing_node_select

    if st.button("🔄 Execute Parallel Superstep"):
        parallel_nodes = ["node_query_vector_db", "node_fetch_user_profile", "node_calculate_risk"]
        p_res = sim.simulate_parallel_superstep(parallel_nodes, selected_fail)
        
        if p_res["superstep_committed"]:
            st.success(f" {p_res['explanation']}")
        else:
            st.error(f"❌ {p_res['explanation']}")

        df_parallel = pd.DataFrame([
            {"Node": k, "Outcome": v["status"], "State Committed": v["committed"]}
            for k, v in p_res["node_statuses"].items()
        ])
        st.dataframe(df_parallel, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: Why is idempotency critical when designing LangGraph nodes with automated retries?", expanded=False):
    st.markdown("""
    **Answer**:
    Because retries and worker crashes can trigger multiple invocations of the same node function.
    If the node performs side-effects (e.g. charging a payment card or writing to an external database) without idempotency tokens, customers will be double-billed or duplicate records will be created.
    Idempotency guarantees that executing the node multiple times produces the exact same result as executing it once.
    """)

with st.expander("Q2: What is the purpose of the checkpoint_writes buffer in the Pregel execution engine?", expanded=False):
    st.markdown("""
    **Answer**:
    `checkpoint_writes` is a temporary staging area that stores pending state updates yielded by active nodes during a superstep.
    Updates remain in `checkpoint_writes` until all parallel nodes in that superstep complete successfully.
    Only then is the atomic barrier crossed, and the writes are committed into the persistent checkpointer.
    """)

with st.expander("Q3: Does LangGraph provide at-least-once or exactly-once execution guarantees?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph provides **at-least-once** node execution guarantees across distributed worker restarts.
    State transitions within the graph are atomic at the superstep level, but external side-effects executed inside node functions must be made idempotent by the developer to achieve **effectively-once** semantics.
    """)

with st.expander("Q4: What happens to parallel nodes in a superstep if one node crashes?", expanded=False):
    st.markdown("""
    **Answer**:
    The superstep acts as an atomic barrier.
    If any node in the superstep throws an unhandled exception, none of the outputs from the peer nodes in that superstep are committed to the checkpointer.
    All staged writes for that superstep are discarded.
    """)

with st.expander("Q5: How can a production team resume a thread that failed halfway through execution?", expanded=False):
    st.markdown("""
    **Answer**:
    By configuring the application with a persistent checkpointer (e.g., `PostgresSaver`), inspecting the failed thread state with `app.get_state(config)`, and then calling `app.invoke(None, config=config)`.
    LangGraph reads `snapshot.next` and restarts execution from the uncommitted step.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Non-Idempotent Webhooks Retrying on Network Drops")
    st.markdown("""
    *Mechanism*: Sending an HTTP webhook to an external ERP system inside a node without a deduplication key.  
    *Consequence*: If the ACK response times out, the node retries and triggers duplicate orders in the ERP.  
    *Fix*: Always pass a unique `Idempotency-Key` HTTP header derived from `thread_id` and superstep index.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Committing State Mutations Directly in Node Side-Effects")
    st.markdown("""
    *Mechanism*: Directly updating external database tables from inside a node before the superstep barrier commits.  
    *Consequence*: If a peer node in the same superstep crashes, LangGraph rolls back state, but the external database remains mutated.  
    *Fix*: Isolate external side-effects into dedicated final nodes or use two-phase commit patterns.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Assuming Thread-Safe Global State in Multi-Worker Environments")
    st.markdown("""
    *Mechanism*: Using Python module-level global variables (`my_cache = {}`) to store deduplication keys across requests.  
    *Consequence*: Multi-pod Kubernetes deployments have isolated RAM; deduplication keys are not shared across workers.  
    *Fix*: Store idempotency tokens in Redis or the checkpointer database.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Reliability Warning): Unbounded Retry Policies Overwhelming Degraded Services")
    st.markdown("""
    *Mechanism*: Setting `max_attempts=10` with zero jitter on a node communicating with an overloaded database.  
    *Consequence*: Multiplies traffic volume exponentially, worsening the outage for all system services.  
    *Fix*: Set `max_attempts <= 3`, use exponential backoff, and always enable jitter.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Idempotency**\\nThe property that executing an operation multiple times produces the exact same result.")
with col_fc2:
    st.info("**checkpoint_writes**\\nThe Pregel staging buffer where pending node outputs wait before atomic commit.")
with col_fc3:
    st.info("**Superstep Barrier**\\nThe atomic synchronization point that commits all parallel node outputs together.")
with col_fc4:
    st.info("**At-Least-Once**\\nThe distributed execution guarantee where tasks are re-run on failure until successful.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the formal definition of idempotency and why it is mandatory for retries.", value=True)
c2 = st.checkbox("I understand the role of the checkpoint_writes staging buffer in Pregel.", value=True)
c3 = st.checkbox("I know what happens when one node crashes during a parallel superstep execution.", value=True)
c4 = st.checkbox("I can differentiate between at-least-once execution and effectively-once semantics.", value=True)
c5 = st.checkbox("I know how to construct deterministic idempotency tokens for external API calls.", value=True)
c6 = st.checkbox("I understand how to inspect pending task writes using snapshot.tasks[0].writes.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 17.3 'Production Reliability' with full architectural and practical comprehension!")
