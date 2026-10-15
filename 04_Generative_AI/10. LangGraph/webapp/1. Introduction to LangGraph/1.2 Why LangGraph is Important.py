import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="1.2 Why LangGraph is Important - LangGraph Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Custom Theme CSS
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
    .badge-approved {
        background-color: #059669;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-paused {
        background-color: #D97706;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE DURABLE AGENT SIMULATOR FOR WORKBENCH & PRODUCTION PIPELINE
# ==============================================================================
class DurableAgentSimulator:
    """
    Simulates LangGraph's durable execution engine with persistent checkpoints
    and human-in-the-loop breakpoints.
    """
    def __init__(self):
        self.checkpoints: Dict[str, List[Dict[str, Any]]] = {}

    def run_step(self, thread_id: str, action: str, amount: float, human_approved: bool = False) -> Dict[str, Any]:
        history = self.checkpoints.get(thread_id, [])
        step_num = len(history) + 1

        if action == "transfer" and amount > 5000 and not human_approved:
            snapshot = {
                "step": step_num,
                "status": "PAUSED_FOR_APPROVAL",
                "action": action,
                "amount": amount,
                "message": f"Transfer of ${amount:,.2f} exceeds threshold. Human sign-off required."
            }
            history.append(snapshot)
            self.checkpoints[thread_id] = history
            return snapshot

        status = "COMPLETED"
        msg = f"Executed {action} for ${amount:,.2f} successfully."
        if human_approved:
            msg += " (Approved by Human Supervisor)"

        snapshot = {
            "step": step_num,
            "status": status,
            "action": action,
            "amount": amount,
            "message": msg
        }
        history.append(snapshot)
        self.checkpoints[thread_id] = history
        return snapshot

    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        return self.checkpoints.get(thread_id, [])


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("1.2 Why LangGraph is Important: Durability, Human-in-the-Loop, & Enterprise Reliability")
st.caption("Chapter 1: Introduction to LangGraph | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
Moving from an AI demonstration to an enterprise-grade autonomous system is not a matter of prompting—it is an 
**infrastructure challenge**. In production, networks drop connections, Kubernetes worker pods are preempted during 
auto-scaling events, high-stakes financial operations require human sign-off, and customer interactions span days.

**LangGraph** is crucial because it provides the **Durable Systems Substrate** for modern AI:
1. **Durable Execution**: Saving exact state at every execution superstep.
2. **Deterministic Human Breakpoints**: Safely pausing execution without active server processes.
3. **Multi-Step Self-Correction**: Enabling models to diagnose and fix errors before returning answers.
4. **Streaming & Observability**: Real-time token streaming and step-by-step audit tracing.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Durable, Fault-Tolerant Enterprise Agent Runtime Lifecycle", divider="rainbow")

st.markdown("""
The architecture below illustrates how LangGraph guarantees 99.99% availability and fault tolerance. 
Every node transition is committed to an external database checkpointer. If a server crashes or an operation requires 
human approval, the execution thread suspends safely and resumes without data loss.
""")

blueprint_dot = """
digraph DurableLifecycle {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_ingress {
        label = "Tier 1: Request Ingress & Thread Locking";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        Client [label="Client / API Request\\n(thread_id = 'org-4821-claim')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Gateway [label="FastAPI Ingress Router\\n(Loads State Snapshot from Checkpointer)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_checkpoint {
        label = "Tier 2: Persistent Storage Layer (Checkpointer)";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        PostgresCheckpointer [label="PostgreSQL / Redis Checkpointer\\n(Durable Commit per Superstep | ACID Compliance)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        BlobStore [label="S3 / GCS Artifact Store\\n(Stores Large PDFs, Audio, & Blobs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_runtime {
        label = "Tier 3: Execution Runtime & Human Breakpoints";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        PlanningNode [label="Superstep 1: Reasoning & Planning\\n(LLM produces action proposal)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        InterruptNode [label="Superstep 2: interrupt() Breakpoint\\n(Pauses graph; notifies human reviewer)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
        ToolExecNode [label="Superstep 3: Tool Execution Node\\n(Executes DB mutations or payments)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_observability {
        label = "Tier 4: Egress & Observability";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#8B5CF6"; fontsize = 11;
        StreamingGateway [label="Streaming Event Hub (SSE)\\n(Emits on_chat_model_stream chunks)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
        LangSmith [label="LangSmith Tracing\\n(Full run tree, latencies, & token costs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#8B5CF6"];
    }

    Client -> Gateway [label="1. Invoke Graph"];
    Gateway -> PostgresCheckpointer [label="2. Load State Snapshot"];
    PostgresCheckpointer -> PlanningNode [label="3. Inject State"];

    PlanningNode -> PostgresCheckpointer [label="4. Commit Superstep 1"];
    PlanningNode -> InterruptNode [label="High-Risk Action Detected"];
    
    InterruptNode -> PostgresCheckpointer [label="5. Suspend State & Save Token"];
    InterruptNode -> Client [label="6. Return HTTP 202 Paused"];

    Client -> InterruptNode [label="7. Resume with Human Approval", color="#EC4899", penwidth=1.8];
    InterruptNode -> ToolExecNode [label="8. Dispatch Action"];
    ToolExecNode -> PostgresCheckpointer [label="9. Commit Superstep 3"];
    ToolExecNode -> StreamingGateway [label="10. Stream Output"];
    StreamingGateway -> LangSmith [label="11. Record Run Trace"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - STATEFUL AI APPLICATIONS
# ==============================================================================
st.header("2. Stateful AI Applications: Explicit Type-Safe State vs Ephemeral Context", divider="rainbow")

st.markdown("""
Stateless APIs (like raw OpenAI `chat.completions`) require clients to re-send the entire chat history on every turn. 
As business applications expand to encompass multi-agent research, document extraction, and multi-user collaboration, 
ephemeral message arrays fall apart.
""")

col_st1, col_st2 = st.columns([1, 1])

with col_st1:
    st.markdown("""
    ### Why Stateful Graphs Are Mandatory
    * **Typed Domain Entities**: Real workflows track business objects (`lead_score: int`, `draft_contract: str`, `approval_status: Enum`), not just strings.
    * **Differential Updates**: Nodes only emit what changed; the graph engine reconciles updates deterministically.
    * **Auditability**: Every modification to every state key is timestamped and recorded for compliance.
    """)

with col_st2:
    st.markdown("""
    ### Type-Safe State Declaration
    """)
    st.code("""
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class EnterpriseState(TypedDict):
    # Channel reducer handles list appending & ID deduplication
    messages: Annotated[List[dict], add_messages]
    claim_id: str
    payout_amount: float
    risk_level: str
    requires_manager_review: bool
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - LONG-RUNNING AGENTS
# ==============================================================================
st.header("3. Long-Running Agents: Thread Suspension, Multi-Day Lifecycles, and Resumption", divider="rainbow")

st.markdown("""
Autonomous agents frequently perform tasks that take hours or days: awaiting asynchronous batch web scrapers, 
waiting for email responses, or queuing for human approval. 

In standard software, keeping a process waiting for hours wastes server CPU/RAM and crashes if the host reboots. 
LangGraph decouples **execution time** from **process uptime**:
* When a task cannot proceed immediately, LangGraph saves the state to PostgreSQL and exits the Python runtime.
* Server resources are instantly freed (0% CPU, 0 MB RAM).
* Days later, an incoming webhook triggers `graph.invoke(Command(resume=payload), config={"configurable": {"thread_id": "..."}})`. 
  The graph re-hydrates the state in 10 milliseconds and resumes execution from that exact step!
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - MULTI-STEP REASONING WORKFLOWS
# ==============================================================================
st.header("4. Multi-Step Reasoning Workflows: Reflection, ReAct Loops, and Dynamic Routing", divider="rainbow")

st.markdown("""
Complex problems require iterative reasoning. In traditional software, if an LLM generates a SQL query with a syntax 
error, the application crashes and returns an error message to the user.
""")

col_rw1, col_rw2 = st.columns([1, 1])

with col_rw1:
    st.markdown("""
    ### The Multi-Step Reflection Pattern
    In LangGraph, self-correction is represented as a natural cyclic edge:
    1. **Plan / Generate**: LLM drafts code or SQL.
    2. **Execute**: Node executes code in a sandbox.
    3. **Evaluate**: Node runs unit tests or linters.
    4. **Reflect & Self-Correct**: If execution yields an error, control routes *backward* to the LLM node with the error traceback, looping until success.
    """)

with col_rw2:
    st.markdown("""
    ### ReAct Loop vs LangGraph State Machine
    * **ReAct Prompting (Legacy)**: Single massive prompt attempting to Reason, Act, and Observe inside one unconstrained text block.
    * **LangGraph State Machine (Modern)**: Rigidly separated nodes for Reasoning and Tool Execution with explicit typed schemas, eliminating format hallucination.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - AGENT ORCHESTRATION
# ==============================================================================
st.header("5. Agent Orchestration: Multi-Node Coordination, Fan-Out, and Fan-In Joins", divider="rainbow")

st.markdown("""
LangGraph provides native primitives for parallel execution and synchronization:
""")

fan_dot = """
digraph FanOutFanIn {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.2", nodesep="0.3", ranksep="0.4"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
    edge [fontname="Helvetica", fontsize=8, color="#64748B"];

    Supervisor [label="Supervisor Agent\\n(Decomposes prompt into 3 research tasks)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    Worker1 [label="Worker 1: Market Analysis", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    Worker2 [label="Worker 2: Financial Metrics", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    Worker3 [label="Worker 3: Legal Risks", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    Synthesizer [label="Synthesizer Node (Fan-In Join)\\n(Aggregates all 3 reports into executive brief)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];

    Supervisor -> Worker1 [label="Parallel Fan-Out (Send API)", color="#10B981"];
    Supervisor -> Worker2 [label="Parallel Fan-Out", color="#10B981"];
    Supervisor -> Worker3 [label="Parallel Fan-Out", color="#10B981"];

    Worker1 -> Synthesizer [label="Merge Channel"];
    Worker2 -> Synthesizer [label="Merge Channel"];
    Worker3 -> Synthesizer [label="Merge Channel"];
}
"""
st.graphviz_chart(fan_dot, use_container_width=True)

st.markdown("""
* **Dynamic Fan-Out (The `Send` API)**: Allows a node to dynamically spawn an arbitrary number of parallel worker nodes based on runtime state (e.g. one worker per chapter or document).
* **Deterministic Fan-In Join**: The `Synthesizer` node only fires once *all* spawned workers have completed their superstep and committed to the shared state channel.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - HUMAN-IN-THE-LOOP SYSTEMS
# ==============================================================================
st.header("6. Human-in-the-Loop Systems: Breakpoints, Approvals, and Time-Travel Editing", divider="rainbow")

st.markdown("""
Autonomous AI should never be allowed to execute destructive or high-stakes actions (sending wire transfers, deleting databases, 
publishing public blog posts) without explicit human confirmation.
""")

col_hitl1, col_hitl2 = st.columns([1, 1])

with col_hitl1:
    st.markdown("""
    ### The Three HITL Modalities
    1. **Approval Gate (Wait for Sign-off)**: Graph halts before executing a critical tool. Human approves or rejects.
    2. **State Modification (Human Editing)**: A human inspects the agent's drafted email or SQL query, edits the text directly in the state snapshot, and resumes.
    3. **Time-Travel (Forking History)**: The human rewinds the thread back 3 steps to alter an earlier prompt decision, generating an alternative execution timeline.
    """)

with col_hitl2:
    st.markdown("""
    ### Native `interrupt()` Mechanism
    """)
    st.code("""
from langgraph.types import interrupt, Command

def refund_node(state):
    if state["refund_amount"] > 1000:
        # Pauses graph execution cleanly!
        approval = interrupt({
            "action": "refund",
            "amount": state["refund_amount"],
            "recipient": state["customer_id"]
        })
        if not approval["is_approved"]:
            return {"status": "REJECTED"}
            
    execute_refund(state["refund_amount"])
    return {"status": "PROCESSED"}
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - DURABLE EXECUTION
# ==============================================================================
st.header("7. Durable Execution: Checkpoint Commit Protocols and Pod Eviction Recovery", divider="rainbow")

st.markdown("""
In enterprise cloud environments (AWS EKS, Google Cloud GKE), worker containers are routinely killed due to 
Spot instance terminations, rolling deploys, and memory pressure. 

**Without Durable Execution**: A container crash kills all in-flight agent tasks. A customer midway through a 15-minute 
data analysis job loses all progress and receives an HTTP 502 error.

**With LangGraph Durable Execution**:
* At the conclusion of every single superstep, the complete state snapshot is persisted to a database checkpointer.
* If container pod #1 dies during superstep 4, the load balancer routes the retry to pod #2.
* Pod #2 queries PostgreSQL for `thread_id`, reloads the state snapshot from superstep 3, and continues without missing a beat!
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - PERSISTENT MEMORY
# ==============================================================================
st.header("8. Persistent Memory: Short-Term Thread State vs Long-Term Cross-Thread Stores", divider="rainbow")

st.markdown("""
LangGraph makes a strict, foundational distinction between **Short-Term Memory** and **Long-Term Memory**:
""")

mem_comp = {
    "Memory Dimension": ["Storage Location", "Lifespan", "Scope", "Access Pattern", "LangGraph API"],
    "Short-Term Working Memory": [
        "Checkpointer (Redis / PostgreSQL)",
        "Single conversational thread",
        "Scoped strictly to one `thread_id`",
        "Loaded automatically at every node step",
        "`StateGraph(State)`"
    ],
    "Long-Term Episodic Memory": [
        "Cross-thread Store (PostgreSQL / Vector DB)",
        "Months to years across user lifespan",
        "Scoped globally to `user_id` or `org_id`",
        "Queried explicitly (semantic recall or key-value)",
        "`BaseStore` (e.g. `store.get(('users', id))`)"
    ]
}
st.dataframe(pd.DataFrame(mem_comp), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TOPIC 8 - STREAMING EXECUTION
# ==============================================================================
st.header("9. Streaming Execution: Real-Time Token Streaming, Node Updates, and Event Hooks", divider="rainbow")

st.markdown("""
Multi-agent systems executing dozens of tool steps take 10 to 60 seconds to complete. If the frontend displays 
nothing until the entire graph finishes, users assume the system has crashed. 

LangGraph provides **Multiple Granular Streaming Protocols**:
""")

col_str1, col_str2 = st.columns([1, 1])

with col_str1:
    st.markdown("""
    ### The Three Streaming Modes
    1. **`values` Mode**: Streams the entire state snapshot at the conclusion of every superstep.
    2. **`updates` Mode**: Streams only the specific delta dictionaries emitted by completed nodes.
    3. **`messages` / `astream_events` Mode**: Streams individual LLM completion tokens the microsecond they are generated inside any node, delivering instantaneous perceived response times!
    """)

with col_str2:
    st.markdown("""
    ### Streaming Syntax
    """)
    st.code("""
# Real-time token streaming from inside graph nodes
async for event in graph.astream_events(
    {"messages": [("user", "Draft report")]},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"].content
        print(chunk, end="", flush=True)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 10: TOPIC 9 - FAULT TOLERANCE
# ==============================================================================
st.header("10. Fault Tolerance: Node Retries, Error Boundaries, and Fallback Routing", divider="rainbow")

st.markdown("""
Third-party APIs and foundation models fail frequently. LangGraph provides multi-level resilience engineering:
""")

col_ft1, col_ft2 = st.columns([1, 1])

with col_ft1:
    st.markdown("""
    ### Automatic Node Retries
    Developers can decorate individual nodes with customized retry policies:
    """)
    st.code("""
from langgraph.pregel import RetryPolicy

# Configure exponential backoff on specific node
builder.add_node(
    "call_search_api",
    search_node,
    retry=RetryPolicy(
        max_attempts=4,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=(ConnectionError, TimeoutError)
    )
)
    """, language="python")

with col_ft2:
    st.markdown("""
    ### Dynamic Error Fallback Routing
    If a node exhausts its retry budget:
    * The node catches the exception and returns `{"error": str(err), "status": "FAILED"}` in state.
    * A conditional edge evaluates the error and routes traffic to a **Fallback Node** (e.g., secondary cloud provider or human escalation) rather than crashing the thread.
    """)

st.divider()

# ==============================================================================
# SECTION 11: TOPIC 10 - PRODUCTION AI SYSTEMS
# ==============================================================================
st.header("11. Production AI Systems: Observability, LangSmith Tracing, and Enterprise SLAs", divider="rainbow")

st.markdown("""
In enterprise production, debugging autonomous multi-agent systems requires complete, sub-millisecond observability.
""")

col_po1, col_po2 = st.columns([1, 1])

with col_po1:
    st.markdown("""
    ### LangSmith Deep Tracing Integration
    * **Zero-Code Setup**: Setting environment variables (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`) automatically instruments every node, edge, and token.
    * **Full Execution Run Trees**: Inspect the exact input and output of every superstep, identify slow nodes, and isolate token consumption.
    * **State Timeline Replay**: Step backward and forward through graph execution to see how state evolved over time.
    """)

with col_po2:
    st.markdown("""
    ### LangGraph Studio
    * A dedicated local desktop visual IDE for testing and debugging LangGraph applications.
    * Allows engineers to view live graph topologies, manually trigger nodes, inspect checkpoints, and inject state edits during interactive debugging sessions.
    """)

st.divider()

# ==============================================================================
# SECTION 12: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("12. Production Implementation: Resilient Enterprise Agent Pipeline with Checkpointing", divider="rainbow")

st.markdown("""
Below is an enterprise-grade, self-contained Python implementation showing durable checkpointer initialization, 
state reduction with `add_messages`, human breakpoint definition, and thread isolation.
""")

prod_code = """
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. State Schema with message channel reducer
class LoanApprovalState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    applicant_id: str
    loan_amount: float
    credit_score: int
    approval_status: str

# 2. Node Functions
def credit_check_node(state: LoanApprovalState) -> Dict[str, Any]:
    score = 720  # Simulated credit score
    return {"credit_score": score}

def risk_evaluation_node(state: LoanApprovalState) -> Dict[str, Any]:
    if state["loan_amount"] > 50000 and state["credit_score"] < 750:
        return {"approval_status": "REQUIRES_SUPERVISOR_SIGN_OFF"}
    return {"approval_status": "AUTO_APPROVED"}

# 3. Router Edge
def route_approval(state: LoanApprovalState) -> str:
    if state["approval_status"] == "REQUIRES_SUPERVISOR_SIGN_OFF":
        return "human_review"
    return "finalize"

def human_review_node(state: LoanApprovalState) -> Dict[str, Any]:
    # In production, this halts via interrupt()
    return {"approval_status": "APPROVED_BY_EXECUTIVE"}

def finalize_node(state: LoanApprovalState) -> Dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": f"Loan {state['approval_status']}"}]}

# 4. Graph Construction with Checkpointer
builder = StateGraph(LoanApprovalState)
builder.add_node("credit_check", credit_check_node)
builder.add_node("risk_evaluation", risk_evaluation_node)
builder.add_node("human_review", human_review_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "credit_check")
builder.add_edge("credit_check", "risk_evaluation")
builder.add_conditional_edges("risk_evaluation", route_approval, {
    "human_review": "human_review",
    "finalize": "finalize"
})
builder.add_edge("human_review", "finalize")
builder.add_edge("finalize", END)

# Attach persistent in-memory checkpointer
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 13: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("13. Interactive Workbench: Durable Execution & Human Breakpoint Simulator", divider="rainbow")

st.markdown("""
Test durable execution and human-in-the-loop approvals. Submit high-value financial actions and observe how the engine 
halts at a persistent checkpoint, waits for human sign-off, and resumes cleanly without state corruption.
""")

if "durable_agent" not in st.session_state:
    st.session_state.durable_agent = DurableAgentSimulator()

agent: DurableAgentSimulator = st.session_state.durable_agent

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Trigger Agent Action")
    
    thread_id = st.text_input("Thread Session ID (`thread_id`)", value="thread-acme-9921")
    action_type = st.selectbox("Action Type", ["query_balance", "transfer", "update_address"], index=1)
    transfer_amount = st.slider("Transaction Amount ($)", min_value=100.0, max_value=25000.0, value=7500.0, step=500.0)
    
    st.caption("Policy: Any transfer exceeding $5,000.00 triggers an automated `interrupt()` breakpoint.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        exec_btn = st.button("Submit Agent Action ➡️", use_container_width=True)
    with col_btn2:
        human_approve_btn = st.button("Approve Pending Action ✅", use_container_width=True)

    if exec_btn:
        result = agent.run_step(thread_id, action_type, transfer_amount, human_approved=False)
        st.rerun()

    if human_approve_btn:
        result = agent.run_step(thread_id, action_type, transfer_amount, human_approved=True)
        st.rerun()

with col_wb2:
    st.subheader("2. Persistent Thread Checkpoint History")
    
    history = agent.get_thread_history(thread_id)
    if history:
        current_status = history[-1]["status"]
        if current_status == "PAUSED_FOR_APPROVAL":
            badge_html = "<span class='badge-paused'>PAUSED (Awaiting Human Approval)</span>"
        else:
            badge_html = "<span class='badge-approved'>COMPLETED</span>"

        st.markdown(f"**Thread Status**: {badge_html}", unsafe_allow_html=True)
        st.metric("Total Checkpoints Committed", len(history))

        st.markdown("#### Checkpoint Ledger")
        for cp in reversed(history):
            with st.container():
                st.markdown(f"**Step {cp['step']}** | Status: `{cp['status']}`")
                st.info(cp["message"])
                st.divider()
    else:
        st.info("No checkpoint history yet for this thread. Submit an action above.")

st.divider()

# ==============================================================================
# SECTION 14: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering why LangGraph is important requires understanding deep production resilience, state snapshotting, 
and concurrency models.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: Why is durable execution mandatory for enterprise AI agents compared to standard web requests?"):
    st.markdown("""
    **Answer**: Standard web requests are ephemeral: if a network drops or a pod restarts, the client retries a simple stateless call. 
    In contrast, enterprise AI agents perform long-running, multi-step actions (researching documents, calling multiple internal APIs, 
    awaiting human approvals) that can span minutes, hours, or days. Without durable execution and step-level checkpointing, any container 
    failure destroys active computation, burning API token costs and leaving downstream databases in inconsistent, partially-mutated states.
    """)

with st.expander("Q2: How does LangGraph achieve Human-in-the-Loop without blocking server threads in memory?"):
    st.markdown("""
    **Answer**: When an agent hits an `interrupt()` breakpoint or a human-review node, LangGraph serializes the full state dictionary 
    to a persistent database checkpointer (PostgreSQL/Redis) tagged with a unique `thread_id`. The execution engine then cleanly halts 
    and exits the Python process. Server memory and CPU are instantly freed. When a human supervisor submits approval via an API, 
    the gateway invokes the graph passing `Command(resume=...)`, loading the exact state snapshot from PostgreSQL and resuming execution.
    """)

with st.expander("Q3: What is the difference between LangGraph's Short-Term Memory and Long-Term Memory?"):
    st.markdown("""
    **Answer**:
    * **Short-Term Memory** is thread-scoped: it persists the sequential dialogue and working variables of an active interaction under a single `thread_id`. It is managed by Checkpointers and is automatically loaded on each superstep.
    * **Long-Term Memory** is cross-thread and user-scoped: it stores semantic facts (user preferences, past conversation summaries, extracted entities) across weeks or months. It is stored in a `BaseStore` (or vector database) and must be explicitly queried by nodes.
    """)

with st.expander("Q4: How does LangGraph's `Send` API differ from standard node edges?"):
    st.markdown("""
    **Answer**: Standard node edges define static control flow between pre-declared nodes. The `Send` API enables **dynamic map-reduce fan-out**: 
    a node can inspect state and dynamically return a list of `Send(node_name, custom_state)` objects. For example, if a document contains 
    12 chapters, a node can spawn 12 parallel executions of a worker node concurrently, each with its own isolated chapter slice, before 
    joining at a downstream synthesis node.
    """)

with st.expander("Q5: What are the primary streaming modes in LangGraph and when should each be used?"):
    st.markdown("""
    **Answer**:
    1. **`values`**: Returns the full state dictionary after each superstep. Best for UI state inspectors and audit dashboards.
    2. **`updates`**: Returns only the delta dictionary emitted by the specific node that just finished. Best for event-driven backend microservices.
    3. **`astream_events` (v2)**: Streams fine-grained internal events, including raw LLM token deltas as they are sampled. Mandatory for interactive chatbots requiring sub-400ms perceived Time-to-First-Token (TTFT).
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Long-Running Async Operations Inside Synchronous Nodes (CRITICAL)**
    * *Error*: Executing blocking network requests or `time.sleep()` inside synchronous graph nodes in high-concurrency environments.
    * *Impact*: Starves the Python GIL, blocks the event loop, and causes massive latency spikes across all concurrent threads.
    * *Fix*: Always use async node functions (`async def`) and `ainvoke()` for production microservices.
    """)

    st.error("""
    **Trap 2: Relying on In-Memory `MemorySaver` in Distributed Production Clusters (CRITICAL)**
    * *Error*: Deploying multiple Kubernetes replicas using the default `MemorySaver()` checkpointer.
    * *Impact*: Because state is saved in local process RAM, requests routed to a different pod fail with 'Thread ID not found'.
    * *Fix*: Use `PostgresSaver` with connection pooling for all distributed multi-replica deployments.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Unbounded State Growth Degrading Database Checkpoint Performance (WARNING)**
    * *Warning*: Storing huge JSON payloads, raw document embeddings, or hundreds of images directly in graph state.
    * *Impact*: Checkpointer writes slow from 5ms to 800ms per superstep, overloading PostgreSQL write bandwidth.
    * *Fix*: Store heavy blobs in S3 or Object Storage; store only lightweight URIs in graph state.
    """)

    st.warning("""
    **Trap 4: Forgetting Idempotency on Resumed Human Breakpoints (WARNING)**
    * *Warning*: Resuming an interrupted graph that executes external mutations without transaction keys.
    * *Impact*: Network retries can accidentally execute payments or database updates twice.
    * *Fix*: Enforce idempotency keys (`idempotency_token = state['id']`) on all side-effecting tools.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: Durable Execution</strong><br><br>
        <em>Front:</em> What is durable execution in LangGraph?<br><br>
        <em>Back:</em> Automatic state persistence at each superstep, enabling recovery from crashes without data loss.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: HITL</strong><br><br>
        <em>Front:</em> How does HITL preserve server resources?<br><br>
        <em>Back:</em> By checkpointing state to DB and exiting the Python process until human resume signal arrives.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: Send API</strong><br><br>
        <em>Front:</em> What is the primary use case of the Send API?<br><br>
        <em>Back:</em> Dynamic parallel map-reduce fan-out to spawn arbitrary worker nodes at runtime.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Checkpointer</strong><br><br>
        <em>Front:</em> Which checkpointer is required for multi-pod Kubernetes clusters?<br><br>
        <em>Back:</em> <code>PostgresSaver</code> (centralized relational database store with connection pooling).
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 15: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("15. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of why LangGraph is essential in production:")

chk1 = st.checkbox("I understand why stateful AI applications require explicit typed state schemas instead of ephemeral message arrays.", value=True)
chk2 = st.checkbox("I can explain how LangGraph executes multi-day long-running agents without consuming server RAM.", value=True)
chk3 = st.checkbox("I know how to design cyclic self-correction loops to repair code or tool errors dynamically.", value=True)
chk4 = st.checkbox("I understand how human-in-the-loop interrupts (`interrupt()`) safely pause and resume execution threads.", value=True)
chk5 = st.checkbox("I can explain the difference between thread-scoped short-term memory and cross-thread long-term memory.", value=True)
chk6 = st.checkbox("I know why `PostgresSaver` is mandatory over `MemorySaver` in distributed multi-pod production environments.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 1.2: Why LangGraph is Important! Proceed to Lesson 1.3: LangGraph Ecosystem.")
