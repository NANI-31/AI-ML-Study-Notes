import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="17.1 Durable Execution - LangGraph Curriculum",
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
    .badge-durable {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-crash {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
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
class DurableExecutionSimulator:
    """
    Simulates durable workflow lifecycles:
    - Normal superstep execution
    - Crash simulation (pod kill)
    - Checkpoint recovery via app.invoke(None, config)
    """
    def __init__(self):
        self.state_history = []
        self.current_status = "READY"

    def run_step(self, action: str) -> Dict[str, Any]:
        if action == "EXECUTE_NODE_1":
            self.current_status = "NODE_1_COMMITTED"
            self.state_history = [
                {"step": 1, "node": "ingest_data", "checkpoint_id": "ckpt_001", "status": "COMMITTED", "data": "10k documents parsed"}
            ]
        elif action == "EXECUTE_NODE_2":
            self.current_status = "NODE_2_COMMITTED"
            self.state_history.append(
                {"step": 2, "node": "generate_embeddings", "checkpoint_id": "ckpt_002", "status": "COMMITTED", "data": "10k vectors generated"}
            )
        elif action == "CRASH_WORKER":
            self.current_status = "CRASHED"
            self.state_history.append(
                {"step": 3, "node": "publish_index", "checkpoint_id": "NONE", "status": "FAILED (SIGKILL)", "data": "Out-of-Memory Pod Eviction"}
            )
        elif action == "RECOVER_AND_RESUME":
            self.current_status = "RECOVERED_AND_COMPLETED"
            # Remove failed step and re-run step 3 cleanly
            self.state_history = [s for s in self.state_history if "FAILED" not in s["status"]]
            self.state_history.append(
                {"step": 3, "node": "publish_index", "checkpoint_id": "ckpt_003", "status": "COMMITTED", "data": "Index successfully published"}
            )

        return {
            "status": self.current_status,
            "steps": self.state_history
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Durable Execution Architecture and Checkpoint State Recovery", divider="rainbow")

st.markdown("""
**Durable Execution** is the distributed systems pattern that treats running code as a resilient,
fault-tolerant state machine.
In LangGraph, node computations run on transient compute workers (Kubernetes pods, serverless containers),
but the **Graph State** is durably committed to an external database checkpointer at every **superstep**.
If compute infrastructure crashes, the runtime cold-boots a fresh worker, re-hydrates the state from the last
checkpoint, and resumes execution strictly from the pending nodes with zero data loss.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    durable_blueprint_dot = """
    digraph DurableArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        graph [splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        CLIENT_INVOKE [label="Client Starts Long Workflow\\napp.invoke(input, config={'configurable': {'thread_id': 'job_104'}})", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_superstep1 {
            label = "Superstep 1: Data Ingest";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            NODE_1 [label="Node 1: Extract\\nExecutes in Pod A", fillcolor="#064E3B", color="#34D399"];
            CKPT_1 [label="PostgresSaver Commit\\nCheckpoint 1 persisted", fillcolor="#0F172A", color="#F59E0B"];
            NODE_1 -> CKPT_1;
        }

        subgraph cluster_superstep2 {
            label = "Superstep 2: Embedding Generation";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            NODE_2 [label="Node 2: Embed Documents\\n🚨 POD A TERMINATED (Spot Instance Eviction)", fillcolor="#3B1A1A", color="#F87171"];
        }

        subgraph cluster_recovery {
            label = "Durable Recovery Engine (Pod B Spawned)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#34D399";
            fontcolor = "#34D399";

            COLD_BOOT [label="Pod B Cold Boot: app.invoke(None, config)\\n1. Reads Checkpoint 1 from Postgres\\n2. Identifies pending 'next': ['Node 2']", fillcolor="#2E1065", color="#C084FC"];
            NODE_2_RETRY [label="Node 2: Embed Documents (Replay)\\nRe-executed cleanly with exact state", fillcolor="#064E3B", color="#34D399"];
            CKPT_2 [label="PostgresSaver Commit\\nCheckpoint 2 persisted", fillcolor="#0F172A", color="#F59E0B"];

            COLD_BOOT -> NODE_2_RETRY;
            NODE_2_RETRY -> CKPT_2;
        }

        CLIENT_INVOKE -> NODE_1;
        CKPT_1 -> NODE_2;
        NODE_2 -> COLD_BOOT [label="Worker Crash Recovery", style=dashed, color="#F87171"];
    }
    """
    st.graphviz_chart(durable_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Durable Principle:</b> Compute is ephemeral, but State is eternal.
Your agents can run for 30 seconds or 30 days; server restarts, pod evictions, and deployment updates
do not interrupt long-running multi-agent workflows.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: WHAT IS DURABLE EXECUTION?
# ==============================================================================
st.header("2. What is Durable Execution?: Decoupling State from Ephemeral Compute", divider="rainbow")

st.markdown("""
In traditional programming, executing an algorithm keeps call stacks, local variables, and object references
in CPU registers and volatile RAM. If the operating system kills the process, all progress is destroyed.

In **Durable Execution**, the workflow engine treats execution as a sequence of atomic transactions:
1. Every superstep barrier commits the channel values to a persistent database.
2. The compute worker can die at any microsecond.
3. A replacement worker recovers by querying `checkpoints` where `thread_id = current_thread` and resumes execution.
""")

# ==============================================================================
# SECTION 3: WHY AGENTS NEED DURABILITY
# ==============================================================================
st.header("3. Why Agents Need Durability: Non-Deterministic Latencies, Outages & Resource Preservation", divider="rainbow")

st.markdown("""
AI agents are especially prone to operational failure:
1. **Long Duration**: Multi-step ReAct loops, web scraping, and deep research tasks take minutes or hours.
2. **Third-Party Rate Limits**: OpenAI, Anthropic, or external APIs return HTTP 429 or 503 errors mid-workflow.
3. **Infrastructure Volatility**: Cloud spot instances are terminated with 30 seconds notice.
4. **Human Invalidation**: A human review gate may take 48 hours to be answered.
""")

# ==============================================================================
# SECTION 4: LONG-RUNNING WORKFLOWS
# ==============================================================================
st.header("4. Long-Running Workflows: Multi-Hour Background Research, Batches & Polling", divider="rainbow")

st.markdown("""
Durable execution enables long-running workflows that sleep, poll external APIs, or await webhooks
without occupying server memory:
""")

st.code("""
# A long-running research agent that can safely span hours
config = {"configurable": {"thread_id": "research_job_9921"}}

# Step 1: Ingest query
app.invoke({"topic": "Quantum Computing Advances"}, config=config)

# 6 hours later, an external webhook notifies the server of new data:
# Resumes execution without restarting from the beginning!
app.invoke({"new_web_data": [...]}, config=config)
""", language="python")

# ==============================================================================
# SECTION 5: RESUME AFTER FAILURE
# ==============================================================================
st.header("5. Resume After Failure: Cold Boot Hydration and Deterministic Node Replay", divider="rainbow")

st.markdown("""
When a server crash occurs, how do you command LangGraph to resume?
Pass `None` as the input argument:
""")

st.code("""
# Passing None tells LangGraph to inspect the pending 'next' nodes
# in the latest checkpoint and resume execution automatically:
resumed_result = app.invoke(None, config={"configurable": {"thread_id": "job_104"}})
""", language="python")

# ==============================================================================
# SECTION 6: CHECKPOINT RECOVERY
# ==============================================================================
st.header("6. Checkpoint Recovery: Loading Pending Tasks from Checkpointer next Tuples", divider="rainbow")

st.markdown("""
How does LangGraph know what to execute when recovering from disk?
Every checkpoint stores the `next` attribute: a tuple containing the names of the nodes
scheduled to run in the subsequent superstep:
""")

st.code("""
state = app.get_state(config)

print("Current Values:", state.values)
print("Pending Scheduled Nodes (next):", state.next)
# Output: ('generate_embeddings',)
""", language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise Durable Worker Pipeline with Crash Recovery", divider="rainbow")

st.markdown("""
Below is the complete production implementation of a durable pipeline:
""")

st.code("""
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class BatchIngestionState(MessagesState):
    processed_count: int
    stage: str

# 2. Checkpointer
checkpointer = MemorySaver()

# 3. Nodes
def ingest_node(state: BatchIngestionState) -> dict:
    return {
        "processed_count": 500,
        "stage": "INGESTED",
        "messages": [AIMessage(content="500 records ingested.")]
    }

def transform_node(state: BatchIngestionState) -> dict:
    return {
        "processed_count": 1000,
        "stage": "TRANSFORMED",
        "messages": [AIMessage(content="All records transformed and validated.")]
    }

# 4. Assemble Graph
builder = StateGraph(BatchIngestionState)
builder.add_node("ingest", ingest_node)
builder.add_node("transform", transform_node)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "transform")
builder.add_edge("transform", END)

durable_app = builder.compile(checkpointer=checkpointer)

# 5. Invocation with thread_id
cfg = {"configurable": {"thread_id": "batch_thread_441"}}
durable_app.invoke({"processed_count": 0, "messages": [HumanMessage(content="Start batch")]}, config=cfg)

# 6. Verify State
snapshot = durable_app.get_state(cfg)
assert snapshot.values["stage"] == "TRANSFORMED"
""", language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Crash Injection, State Hydration & Recovery Simulator", divider="rainbow")

st.markdown("""
Simulate superstep execution, inject a pod crash, and test cold-boot recovery from the latest
checkpoint. Observe how progress is preserved across worker failures.
""")

sim = DurableExecutionSimulator()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Lifecycle Simulation Actions")
    if st.button("1. Run Step 1: Ingest Data"):
        sim.run_step("EXECUTE_NODE_1")
        st.session_state.durable_sim = sim.state_history
        st.session_state.durable_status = sim.current_status

    if st.button("2. Run Step 2: Generate Vectors"):
        sim.run_step("EXECUTE_NODE_2")
        st.session_state.durable_sim = sim.state_history
        st.session_state.durable_status = sim.current_status

    if st.button("3. 🔥 Simulate Server Crash (SIGKILL)"):
        sim.run_step("CRASH_WORKER")
        st.session_state.durable_sim = sim.state_history
        st.session_state.durable_status = sim.current_status

    if st.button("4. 🔄 Recover and Resume (invoke(None))"):
        sim.run_step("RECOVER_AND_RESUME")
        st.session_state.durable_sim = sim.state_history
        st.session_state.durable_status = sim.current_status

    if st.button("Reset Simulation"):
        st.session_state.durable_sim = []
        st.session_state.durable_status = "READY"
        st.rerun()

with wb_col2:
    st.subheader("📋 Durable State Machine Telemetry")
    curr_status = st.session_state.get("durable_status", "READY")
    badge = "badge-crash" if "CRASHED" in curr_status else "badge-durable"
    st.markdown(f"**Current State**: <span class='{badge}'>{curr_status}</span>", unsafe_allow_html=True)

    history = st.session_state.get("durable_sim", [])
    if history:
        for item in history:
            st.markdown(f"**Step {item['step']} — Node: `{item['node']}`** `[{item['status']}]`")
            st.write(f"- Checkpoint ID: `{item['checkpoint_id']}`")
            st.write(f"- Data Payload: {item['data']}")
            st.divider()
    else:
        st.info("Workflow is ready. Click '1. Run Step 1' to start.")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What is the core definition of durable execution in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Durable execution is the decoupling of workflow state from compute worker lifecycles.
    Graph state is committed to an external database checkpointer at every superstep, ensuring that
    infrastructure crashes, pod evictions, or redeployments can be recovered with zero loss of progress.
    """)

with st.expander("Q2: How do you programmatically command LangGraph to resume after a worker crash?", expanded=False):
    st.markdown("""
    **Answer**:
    Call `app.invoke(None, config=config)` using the same `thread_id`.
    Passing `None` as the input payload commands the engine to look up the latest checkpoint, inspect the scheduled `next` nodes,
    and resume execution from that exact step.
    """)

with st.expander("Q3: Why must checkpointer commits occur at superstep boundaries?", expanded=False):
    st.markdown("""
    **Answer**:
    Supersteps represent synchronized barriers where all active parallel nodes have completed and their return deltas
    have been reconciled by channel reducers. Committing at this boundary guarantees atomic consistency.
    """)

with st.expander("Q4: What is the purpose of the 'next' attribute on a StateSnapshot?", expanded=False):
    st.markdown("""
    **Answer**:
    `state.next` is a tuple indicating which nodes are scheduled to execute in the subsequent superstep.
    During recovery, the engine reads `state.next` to determine which node functions to invoke.
    """)

with st.expander("Q5: Can durable execution handle spot instance evictions gracefully?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! When a spot instance terminates, Kubernetes spins up a new pod on an available node.
    The new pod connects to PostgreSQL, loads the latest checkpoint, and continues the job without human intervention.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Passing Fresh Inputs When Trying to Resume Crashed Threads")
    st.markdown("""
    *Mechanism*: Calling `app.invoke({"input": "retry"}, config=config)` on a crashed thread.  
    *Consequence*: Overwrites or conflicts with the existing message history and state channels.  
    *Fix*: Always pass `app.invoke(None, config=config)` to resume from pending nodes.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Using In-Memory Checkpointers in Distributed Deployments")
    st.markdown("""
    *Mechanism*: Using `MemorySaver` in Kubernetes.  
    *Consequence*: When Pod A crashes, Pod B has an empty RAM store and cannot resume the workflow.  
    *Fix*: Deploy `PostgresSaver` backed by a durable relational database.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Non-Idempotent Database Writes During Recovery")
    st.markdown("""
    *Mechanism*: Re-executing a node that inserts database rows without unique deduplication keys.  
    *Consequence*: Duplicate records inserted if the node crashed halfway through execution.  
    *Fix*: Ensure node side-effects use unique idempotency keys.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Large Binary Payloads in State Checkpoints")
    st.markdown("""
    *Mechanism*: Storing 100MB video or PDF files directly in state channels.  
    *Consequence*: Severely slows down checkpoint serialization and bloats database storage.  
    *Fix*: Store large files in S3/GCS and keep only the object URI in the state channel.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Durable Execution**\\nDecoupling workflow state from compute workers to guarantee zero-loss crash resilience.")
with col_fc2:
    st.info("**app.invoke(None)**\\nThe command used to resume execution from the pending next nodes in the latest checkpoint.")
with col_fc3:
    st.info("**Superstep Barrier**\\nThe atomic synchronization point where parallel nodes complete and state is committed.")
with col_fc4:
    st.info("**StateSnapshot.next**\\nThe tuple specifying which nodes are scheduled to execute in the next step.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the formal definition of durable execution in LangGraph.", value=True)
c2 = st.checkbox("I understand why long-running AI agents require database-backed durability.", value=True)
c3 = st.checkbox("I know how to resume a crashed workflow using app.invoke(None, config=config).", value=True)
c4 = st.checkbox("I understand the role of the StateSnapshot.next tuple during recovery.", value=True)
c5 = st.checkbox("I know how to ensure node idempotency to prevent duplicate side-effects.", value=True)
c6 = st.checkbox("I understand why large binary files should not be stored directly in state channels.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 17.1 'Durable Execution' with full architectural and practical comprehension!")
