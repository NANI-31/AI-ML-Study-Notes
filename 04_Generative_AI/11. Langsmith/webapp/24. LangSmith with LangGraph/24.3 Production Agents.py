import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.3 Production Agents - LangSmith Master Curriculum",
    page_icon="🤖",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ProductionAgentTracingEngine:
    """
    Production Multi-Agent Observability & Human-in-the-Loop (HITL) Dispatcher.
    Simulates enterprise supervisor-worker multi-agent topologies, durable long-running
    execution checkpointers, interrupt() approval gates, state mutations, and real-time
    token/latency telemetry in LangSmith.
    """
    def __init__(self, project_name: str = "enterprise-agent-observability"):
        self.project_name = project_name
        self.recorded_sessions: List[Dict[str, Any]] = []
        self._execution_counter = 5000

    def execute_multi_agent_workflow(
        self,
        thread_id: str,
        workflow_type: str,
        human_decision: str = "approve",
        human_feedback: str = ""
    ) -> Dict[str, Any]:
        """
        Executes a multi-agent workflow featuring Supervisor dispatch, Specialist execution,
        an explicit interrupt() approval gate, and final fulfillment.
        """
        self._execution_counter += 1
        run_id = f"prod_agent_run_{self._execution_counter}"

        # Timeline of agent handoffs and tool executions
        timeline = []
        total_tokens = 0
        total_latency = 0.0

        # Step 1: Supervisor Dispatcher
        step_1_tokens = 450
        step_1_lat = 180.0
        total_tokens += step_1_tokens
        total_latency += step_1_lat
        timeline.append({
            "step": 1,
            "agent": "Supervisor Agent",
            "action": "Plan & Route",
            "details": f"Decomposed workflow '{workflow_type}' into subtasks; assigned to Specialist Agent.",
            "status": "completed",
            "tokens": step_1_tokens,
            "latency_ms": step_1_lat
        })

        # Step 2: Specialist Agent Execution
        step_2_tokens = 1200
        step_2_lat = 420.0
        total_tokens += step_2_tokens
        total_latency += step_2_lat
        timeline.append({
            "step": 2,
            "agent": "Specialist Agent (Engineer/Researcher)",
            "action": "Task Execution & Tool Invocations",
            "details": "Generated patch code, audited dependencies, and verified test suites.",
            "status": "completed",
            "tokens": step_2_tokens,
            "latency_ms": step_2_lat
        })

        # Step 3: HITL Interrupt Gate
        step_3_tokens = 150
        step_3_lat = 45.0
        total_tokens += step_3_tokens
        total_latency += step_3_lat

        interrupt_status = "interrupted"
        if human_decision == "approve":
            interrupt_outcome = "Human APPROVED deployment action without changes."
            next_step_allowed = True
        elif human_decision == "edit":
            interrupt_outcome = f"Human MODIFIED state parameters: '{human_feedback}'."
            next_step_allowed = True
        else: # reject
            interrupt_outcome = f"Human REJECTED action: '{human_feedback}'. Execution halted."
            next_step_allowed = False

        timeline.append({
            "step": 3,
            "agent": "Security & Gatekeeper Node",
            "action": "interrupt() Evaluation Gate",
            "details": f"Triggered interrupt(). Human Decision: {human_decision.upper()}. Details: {interrupt_outcome}",
            "status": "approved" if next_step_allowed else "rejected",
            "tokens": step_3_tokens,
            "latency_ms": step_3_lat
        })

        # Step 4: Deployment / Fulfillment (Only if approved/edited)
        if next_step_allowed:
            step_4_tokens = 380
            step_4_lat = 210.0
            total_tokens += step_4_tokens
            total_latency += step_4_lat
            timeline.append({
                "step": 4,
                "agent": "Deployment Manager Agent",
                "action": "Production Release Fulfillment",
                "details": "Executed blue-green deployment container swap; verified health endpoints.",
                "status": "completed",
                "tokens": step_4_tokens,
                "latency_ms": step_4_lat
            })
            final_status = "completed"
        else:
            final_status = "aborted_by_human"

        session_record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "workflow_type": workflow_type,
            "final_status": final_status,
            "human_decision": human_decision,
            "human_feedback": human_feedback,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "estimated_cost_usd": round((total_tokens / 1000.0) * 0.005, 4),
            "timeline": timeline
        }

        self.recorded_sessions.append(session_record)
        return session_record


# Instantiate engine
prod_agent_tracer = ProductionAgentTracingEngine()

# --- HEADER SECTION ---
st.title("🤖 Chapter 24.3: Production Agents & Multi-Agent Observability")
st.markdown("""
Welcome to the production operations manual for **Autonomous Agents** and **Multi-Agent Systems** traced in **LangSmith**.
Deploying agents to production represents the pinnacle of GenAI engineering complexity.
Production agents run over long horizons, maintain persistent thread-scoped memory, coordinate across multi-agent swarms,
and pause for human-in-the-loop approvals before executing irreversible real-world actions.
This module teaches you how to monitor, debug, and govern enterprise production agents with LangSmith.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Production Multi-Agent Swarm Monitoring & Resilient HITL Architecture", divider="rainbow")

st.markdown("""
The architecture below illustrates the end-to-end telemetry loop of an enterprise multi-agent deployment.
A central Supervisor coordinates specialist worker agents, streams intermediate tool calls to LangSmith,
triggers an explicit `interrupt()` checkpoint before critical operations, ingests human feedback, and resumes seamlessly.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiAgentProduction {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_orchestrator {
            label="1. Top-Level Supervisor & Ingress";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            user_req [label="Client Enterprise Request\nthread_id: 'prod-wf-9941'\nTenant: 'Enterprise-Finance'", fillcolor="#1E293B", color="#38BDF8"];
            supervisor [label="Supervisor Agent Node\nTask Router & Goal Decomposition", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_specialists {
            label="2. Specialized Worker Agents (Swarm)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            research_agent [label="Research Agent\nVectorDB & Web Search", fillcolor="#1E293B", color="#38BDF8"];
            coder_agent [label="Coder / DevOps Agent\nGenerates Terraform / Code Patch", fillcolor="#1E293B", color="#38BDF8"];
            qa_agent [label="QA & Security Auditor\nValidates Policy & CVE Scans", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_hitl {
            label="3. Human-in-the-Loop (HITL) Gate";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            interrupt_node [label="interrupt() Primitive\nHalts Graph Super-Step\nCheckpoint Saved to Postgres", fillcolor="#0F172A", color="#F59E0B"];
            human_ui [label="Human Reviewer Portal\nApprove / Reject / Edit State", fillcolor="#0F172A", color="#A855F7"];
            resume_trigger [label="Resume with Token\napp.invoke(Command(resume=...))", fillcolor="#0F172A", color="#10B981"];
        }

        subgraph cluster_langsmith_monitoring {
            label="4. LangSmith Enterprise Monitoring & Governance";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_multi_trace [label="Multi-Agent Trace Hierarchy\nSupervisor -> Worker -> ToolNode\nMessage Lineage Preserved", fillcolor="#0F172A", color="#38BDF8"];
            ls_dash [label="Real-Time Production Dashboards\nCost/Token Burn, p95 Latency, HITL Latency", fillcolor="#0F172A", color="#10B981"];
            ls_alerts [label="Drift & Infinite Loop Sentry\nAlert: Agent Loops > 8 or Tool Errors > 15%", fillcolor="#7F1D1D", color="#EF4444"];
        }

        user_req -> supervisor;
        supervisor -> research_agent [label="Handoff: Research"];
        supervisor -> coder_agent [label="Handoff: Code"];
        coder_agent -> qa_agent [label="Handoff: Audit"];
        qa_agent -> interrupt_node [label="Approval Required"];

        interrupt_node -> human_ui [label="Pause Execution"];
        human_ui -> resume_trigger [label="Human Signed Off"];
        resume_trigger -> supervisor [label="Resume Execution"];

        supervisor -> ls_multi_trace [label="Top-Level Span"];
        coder_agent -> ls_multi_trace [label="Worker Span"];
        interrupt_node -> ls_multi_trace [label="HITL Span"];
        ls_multi_trace -> ls_dash;
        ls_multi_trace -> ls_alerts;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Six Core Pillars of Production Agent Operations
The following matrix summarizes the engineering operationalization for each syllabus topic:
""")

prod_agent_matrix_df = pd.DataFrame([
    {
        "Operational Topic": "1. Long-Running Agents",
        "Architecture Pattern": "Durable Checkpointing (PostgresSaver / MongoDBSaver)",
        "LangSmith Observability Role": "Tracks multi-day asynchronous runs, resume spans, and cross-process lifecycle events"
    },
    {
        "Operational Topic": "2. Stateful Agents",
        "Architecture Pattern": "Thread-Scoped State Schemas & Managed Memories",
        "LangSmith Observability Role": "Audits state size evolution, channel reducer diffs, and cross-session entity persistence"
    },
    {
        "Operational Topic": "3. Human-in-the-Loop Agents",
        "Architecture Pattern": "LangGraph interrupt() & Command(resume=...)",
        "LangSmith Observability Role": "Captures pause duration, human approval status, state overrides, and reviewer comments"
    },
    {
        "Operational Topic": "4. Multi-Agent Workflows",
        "Architecture Pattern": "Supervisor vs Hierarchical Swarm Topologies",
        "LangSmith Observability Role": "Maps inter-agent delegation spans, message lineage passing, and collective consensus"
    },
    {
        "Operational Topic": "5. Agent Monitoring",
        "Architecture Pattern": "Continuous Telemetry & Metric Aggregations",
        "LangSmith Observability Role": "Tracks token burn rate, cost per completed goal, p95/p99 turn latencies, and tool success rate"
    },
    {
        "Operational Topic": "6. Agent Debugging",
        "Architecture Pattern": "Automated Loop Detection & Triage Alerts",
        "LangSmith Observability Role": "Isolates hallucinated tool arguments, infinite oscillation cycles, and state schema corruption"
    }
])
st.dataframe(prod_agent_matrix_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: LONG-RUNNING AGENTS
# ==============================================================================
st.header("2. Long-Running Agents (Multi-Hour Workflow Checkpointing, Async Resume & Durable Execution)", divider="rainbow")

st.markdown("""
Unlike stateless API endpoints, enterprise agents often run for minutes, hours, or days.
Examples include compiling competitive intelligence dossiers, migrating cloud databases,
or waiting for external webhooks and partner API responses.
""")

st.markdown("### Architecture of Durable Execution")
st.markdown("""
1. **Durable Persistence Engines**: Replacing in-memory checkpointers with persistent storage backends such as `PostgresSaver` or `CosmosDBSaver`.
2. **Asynchronous Resume Semantics**: When waiting for long-running batch jobs, the graph suspends execution. When the external task completes, a webhook invokes `app.invoke()` with the original `thread_id` to resume from the latest checkpoint.
3. **Cooperative Process Resiliency**: If an agent pod in Kubernetes is terminated or scaled down during execution, the newly spun-up pod retrieves the state from Postgres and resumes execution without re-running earlier steps.
""")

st.code('''
from langgraph.checkpoint.postgres import PostgresSaver

# Connect to durable production database
with PostgresSaver.from_conn_string("postgresql://user:pass@db:5432/agents") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
    
    # Start long-running workflow
    config = {"configurable": {"thread_id": "batch-job-7821"}}
    app.invoke({"task": "Reconcile quarterly tax ledgers"}, config=config)
    
    # Process can be terminated here...
    # Resume 4 hours later seamlessly from last saved super-step:
    app.invoke(None, config=config)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith, long-running agent threads are treated as persistent unified sessions:
* Each resumption step appends new runs to the existing trace hierarchy without fragmenting the audit trail.
* LangSmith displays the cumulative elapsed time, active execution duration, and idle waiting duration side-by-side.
""")

st.divider()

# ==============================================================================
# SECTION 3: STATEFUL AGENTS
# ==============================================================================
st.header("3. Stateful Agents (Thread-Scoped Memories, Cross-Session Persistence & Channel Reducer Audits)", divider="rainbow")

st.markdown("""
Stateful agents maintain conversational context and episodic knowledge across multiple user sessions.
In production, state management must balance context continuity against token bloat and privacy compliance.
""")

st.markdown("### Production Memory Architecture")
st.markdown("""
* **Short-Term Working Memory**: Managed via LangGraph state channels (`messages: Annotated[List, add_messages]`), scoped strictly to the current `thread_id`.
* **Long-Term Episodic Memory**: Managed via LangGraph Store (`BaseStore`), allowing agents to persist user preferences (e.g., "User prefers TypeScript over Python") across completely separate threads.
* **Message Trimming & Summarization**: Periodically invoking `trim_messages` or running background summarization nodes to prevent the state channel from exceeding LLM context windows.
""")

st.code('''
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph

# Long-term cross-thread memory store
store = InMemoryStore()

# Writing cross-thread user preferences
user_namespace = ("users", "user_123")
store.put(user_namespace, "preferences", {"language": "python", "framework": "fastapi"})

# Reading in an agent node across any thread
def personalized_node(state, config, *, store):
    user_id = config["configurable"]["user_id"]
    prefs = store.get(("users", user_id), "preferences")
    # Tailor agent response to user preferences...
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith:
* Store operations (`get`, `put`, `search`) appear as instrumented span events in the trace view.
* Engineers can audit how memories were retrieved and verify that sensitive user data complies with GDPR/SOC2 retention policies.
""")

st.divider()

# ==============================================================================
# SECTION 4: HUMAN-IN-THE-LOOP AGENTS
# ==============================================================================
st.header("4. Human-in-the-Loop Agents (interrupt() Spans, Approval Gates, State Edits & Human Feedback Loops)", divider="rainbow")

st.markdown("""
Autonomous agents must not execute high-stakes or irreversible actions (such as deploying code,
executing wire transfers, or deleting databases) without explicit human confirmation.
LangGraph provides the `interrupt()` primitive to implement deterministic human review gates.
""")

st.markdown("### The Modern `interrupt()` Pattern in LangGraph v0.2+")
st.markdown("""
1. **Raising the Interrupt**: A node calls `interrupt({"action": "deploy_prod", "payload": code_diff})`. This suspends the graph execution immediately and saves an interrupted checkpoint.
2. **Reviewer Inspection**: The human reviews the proposed action in an administrative UI or LangSmith portal.
3. **Resuming with Command**: The client resumes the graph using `Command(resume={"approved": True})`. The return value of `interrupt()` becomes the payload passed to the resume command!
4. **State Mutation During Pause**: A human can edit state channels directly using `app.update_state()` before resuming.
""")

st.code('''
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph

def deployment_gate_node(state: AgentState) -> Dict[str, Any]:
    # Interrupt execution and wait for human decision
    review_request = {
        "action": "deploy_container",
        "target_cluster": "production-eu-west",
        "image_tag": state["release_tag"]
    }
    
    # Execution halts here until human resumes!
    human_response = interrupt(review_request)
    
    if not human_response.get("approved"):
        return {"status": "aborted_by_human"}
        
    return {"status": "deployment_in_progress"}

# Client resumes execution after human approval:
app.invoke(Command(resume={"approved": True}), config=thread_config)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith provides native support for LangGraph interrupts:
* When a run hits `interrupt()`, its status in LangSmith changes to `INTERRUPTED`.
* The trace displays the exact payload passed to `interrupt()`.
* When the human resumes the run, LangSmith attaches the human's response and links the subsequent spans to the same audit record.
""")

st.divider()

# ==============================================================================
# SECTION 5: MULTI-AGENT WORKFLOWS
# ==============================================================================
st.header("5. Multi-Agent Workflows (Supervisor vs Swarm Topologies, Inter-Agent Message Lineage & Handoff Traces)", divider="rainbow")

st.markdown("""
As tasks grow in scope, single monolithic agents degrade in reliability.
Enterprise systems decompose responsibilities across specialized agent teams organized in
**Supervisor** or **Swarm / Peer-to-Peer** topologies.
""")

st.markdown("### Top Multi-Agent Architectural Topologies")
st.markdown("""
1. **Supervisor Architecture**: A central routing agent evaluates user requests and delegates subtasks to specialized worker agents (e.g., Coder, Researcher, Reviewer). Workers return outputs to the Supervisor, who decides next steps.
2. **Hierarchical Teams**: Supervisors manage sub-supervisors, creating multi-tier organizational hierarchies for complex software engineering or document generation pipelines.
3. **Swarm / Direct Handoff**: Agents pass control directly to peer agents using explicit handoff tools (`transfer_to_support_agent()`).
""")

st.code('''
def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Evaluates current conversation and delegates to the next specialist agent.
    """
    options = ["researcher", "coder", "reviewer", "FINISH"]
    # LLM structured output selects next destination
    decision = supervisor_model.invoke(state["messages"])
    return {"next_agent": decision.next_worker}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith, multi-agent workflows are visualized as nested span trees:
* The root run captures the overall workflow.
* Each agent delegation appears as a child span containing its local message history.
* Engineers can click between agent spans to audit the exact handoff messages and verify that instructions were transferred accurately without prompt degradation.
""")

st.divider()

# ==============================================================================
# SECTION 6: AGENT MONITORING
# ==============================================================================
st.header("6. Agent Monitoring (Token Burn Accounting, Tool Success Rates, Loop Count & Drift Alerts)", divider="rainbow")

st.markdown("""
Monitoring production agents requires telemetry beyond traditional web server metrics (CPU, RAM).
Engineers must track LLM-specific operational dimensions in real time to prevent runaway expenses
and maintain system reliability.
""")

st.markdown("### The Four Golden Signals of Agent Telemetry")
st.markdown("""
* **1. Token Burn Velocity & Cost**: Total input, output, and cached tokens consumed per user session. Agents consuming > 25,000 tokens per goal trigger cost-anomaly alerts.
* **2. Tool Execution Success Rate**: Percentage of tool calls that return clean payloads. A sudden dip below 90% indicates third-party API deprecations or backend outages.
* **3. Trajectory Step Depth**: Number of node transitions required to complete tasks. Increasing step depth signals agent confusion or prompt drift.
* **4. HITL Approval Latency**: Time elapsed between raising an `interrupt()` and receiving a human response, identifying bottlenecks in human review queues.
""")

st.code('''
# Querying LangSmith API for production agent metrics
from langsmith import Client

client = Client()

# Fetch failed or high-cost runs in production project
problematic_runs = client.list_runs(
    project_name="prod-agent-swarm",
    filter='and(gt(total_tokens, 20000), eq(status, "error"))',
    limit=50
)

for run in problematic_runs:
    print(f"Run {run.id}: Burned {run.total_tokens} tokens with error: {run.error}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams configure automated **LangSmith Monitors & Webhooks**:
* Real-time charts track daily token consumption grouped by agent persona (e.g., Coder vs Researcher).
* Webhook integrations post alerts to Slack and PagerDuty if the average loop count exceeds 5 iterations over a 30-minute window.
""")

st.divider()

# ==============================================================================
# SECTION 7: AGENT DEBUGGING
# ==============================================================================
st.header("7. Agent Debugging (Stuck Agent Loop Detection, Hallucinated Tool Calls & State Corruption Triage)", divider="rainbow")

st.markdown("""
When an agent misbehaves in production, diagnosing root causes requires systematic investigation.
Unlike conventional software with deterministic code paths, agent bugs stem from stochastic reasoning failures,
prompt ambiguity, or subtle edge condition errors.
""")

st.markdown("### Common Production Agent Pathologies & Triage Recipes")
st.markdown("""
1. **The Stuck Loop (Oscillation)**:
   * *Symptom*: Agent alternates endlessly between two tools (e.g., Search -> Calc -> Search -> Calc).
   * *Triage in LangSmith*: Inspect the state diff between cycles. If the tool output is not updating the state channel or if the LLM cannot interpret the tool result, it repeats the query.
2. **Hallucinated Tool Calls**:
   * *Symptom*: Tool execution crashes with `pydantic.ValidationError`.
   * *Triage in LangSmith*: Compare the JSON arguments emitted by the model against the tool's schema. Clarify parameter descriptions or provide few-shot examples in system prompts.
3. **State Corruption / Channel Amnesia**:
   * *Symptom*: Agent asks the user for information already provided three turns ago.
   * *Triage in LangSmith*: Inspect checkpoint message channels. Check whether a node accidentally returned an unannotated dict update that replaced the message history.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith provides **Time-Travel Forking**:
* When an agent produces a corrupted state or gets stuck in a loop, engineers click **"Fork from Checkpoint"** in LangSmith.
* They modify the prompt or adjust state parameters directly in the UI and test the resumed run in a sandboxed environment to verify the fix before pushing code to production.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Production Multi-Agent Monitoring & HITL Dispatcher (ProductionAgentTracingEngine)", divider="rainbow")

st.markdown("""
Below is the production dispatcher implementation.
The `ProductionAgentTracingEngine` models a complete multi-agent workflow featuring
supervisor decomposition, specialist execution, `interrupt()` approval gating, and token/cost tracking.
""")

st.code('''
class ProductionAgentTracingEngine:
    """
    Production Multi-Agent Observability & Human-in-the-Loop (HITL) Dispatcher.
    Simulates enterprise supervisor-worker multi-agent topologies, durable long-running
    execution checkpointers, interrupt() approval gates, and real-time token/latency telemetry.
    """
    def __init__(self, project_name: str = "enterprise-agent-observability"):
        self.project_name = project_name
        self.recorded_sessions: List[Dict[str, Any]] = []
        self._execution_counter = 5000

    def execute_multi_agent_workflow(
        self,
        thread_id: str,
        workflow_type: str,
        human_decision: str = "approve",
        human_feedback: str = ""
    ) -> Dict[str, Any]:
        """
        Executes a multi-agent workflow featuring Supervisor dispatch, Specialist execution,
        an explicit interrupt() approval gate, and final fulfillment.
        """
        # Emulates step-by-step multi-agent execution, interrupt() halts,
        # and resumption with Command(resume=...).
        ...
''', language="python")

st.info("The production engine is instantiated in module scope and powers the interactive multi-agent sandbox below.")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Agent Handoff Simulator & HITL Interrupt Approval Sandbox", divider="rainbow")

st.markdown("""
Experience production multi-agent orchestration and human-in-the-loop governance firsthand.
Configure a mission-critical enterprise workflow, simulate the agent swarm handoff,
and test the `interrupt()` approval gate by approving, rejecting, or modifying the agent's action.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Multi-Agent Workflow Controls")
    workflow_choice = st.selectbox(
        "Select Enterprise Task:",
        options=[
            "Deploy Blue-Green Infrastructure Patch",
            "Execute Financial Portfolio Rebalancing",
            "Generate & Publish Customer Security Advisory"
        ]
    )

    thread_name = st.text_input("Session Thread ID:", value="thread_prod_alpha_921")

    st.markdown("#### Human-in-the-Loop Reviewer Gate")
    hitl_decision = st.radio(
        "Human Reviewer Action at interrupt() Gate:",
        options=["approve", "edit", "reject"],
        format_func=lambda d: {
            "approve": "✅ Approve (Proceed to Fulfillment)",
            "edit": "✏️ Modify State (Add Constraints & Proceed)",
            "reject": "🛑 Reject & Abort (Cancel Execution)"
        }[d]
    )

    feedback_text = ""
    if hitl_decision == "edit":
        feedback_text = st.text_input("State Override Parameters:", value="Require staging smoke test first")
    elif hitl_decision == "reject":
        feedback_text = st.text_input("Rejection Rationale:", value="Security team identified unverified package")

    sim_btn = st.button("🚀 Run Multi-Agent Swarm with HITL Gate", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Multi-Agent Trace & Telemetry")

    if sim_btn or len(prod_agent_tracer.recorded_sessions) == 0:
        session = prod_agent_tracer.execute_multi_agent_workflow(
            thread_id=thread_name,
            workflow_type=workflow_choice,
            human_decision=hitl_decision,
            human_feedback=feedback_text
        )
    else:
        session = prod_agent_tracer.recorded_sessions[-1]

    # Metrics Overview
    m1, m2, m3, m4 = st.columns(4)
    status_badge = "COMPLETED" if session["final_status"] == "completed" else "HALTED (HITL REJECT)"
    m1.metric("Final Status", status_badge)
    m2.metric("Total Tokens", f"{session['total_tokens']:,}")
    m3.metric("Total Latency", f"{session['total_latency_ms']:.1f} ms")
    m4.metric("Est. Cost", f"${session['estimated_cost_usd']:.4f}")

    if session["final_status"] == "completed":
        st.success("🎉 **Workflow Succeeded**: Multi-agent team completed the task and satisfied the human gate.")
    else:
        st.warning("⚠️ **Workflow Aborted**: Execution safely stopped at the human review checkpoint.")

    # Timeline of Agent Handoffs
    st.markdown("#### Agent Execution Waterfall & Handoff Lineage")
    timeline_df = pd.DataFrame(session["timeline"])
    st.dataframe(timeline_df[["step", "agent", "action", "status", "tokens", "latency_ms"]], use_container_width=True)

    # Detailed Step Card
    st.markdown("#### Step-by-Step Audit Breakdown")
    for step in session["timeline"]:
        with st.container(border=True):
            st.markdown(f"**Step {step['step']} — {step['agent']}** | Status: `{step['status']}`")
            st.markdown(f"*Action*: {step['action']}")
            st.markdown(f"*Details*: {step['details']}")
            st.caption(f"Tokens: {step['tokens']} | Latency: {step['latency_ms']} ms")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does LangGraph implement Human-in-the-Loop (HITL) without blocking server threads?"):
    st.markdown("""
    **Answer:**
    LangGraph implements asynchronous HITL via the `interrupt()` primitive:
    1. When a node calls `interrupt(payload)`, LangGraph halts the current super-step and writes an interrupted checkpoint snapshot to durable storage (e.g., Postgres).
    2. The server process terminates the immediate invocation cleanly, freeing CPU and worker threads—no thread is kept sleeping or blocking in memory.
    3. When a human reviewer submits a decision (minutes or days later), the client invokes the graph using `Command(resume=payload)` targeting the original `thread_id`.
    4. LangGraph rehydrates state from the checkpoint and resumes execution seamlessly from the exact line where `interrupt()` was called.
    """)

with st.expander("Q2: What is the architectural difference between a Supervisor Agent topology and a Peer-to-Peer Swarm?"):
    st.markdown("""
    **Answer:**
    * **Supervisor Topology**: Features a centralized coordinator node. Workers report back to the supervisor, who maintains global state and decides the next agent to dispatch.
      * *Pros*: Strict centralized governance, deterministic routing, easier auditing.
      * *Cons*: Supervisor is a latency and token bottleneck.
    * **Peer-to-Peer Swarm**: Agents directly hand off control to other agents via dedicated handoff tools without routing back to a central master.
      * *Pros*: Flexible, lower latency for sequential handoffs.
      * *Cons*: Risk of circular handoff ping-pong and difficult global state coordination.
    """)

with st.expander("Q3: How do you detect and mitigate 'Agent Drift' in long-running production environments?"):
    st.markdown("""
    **Answer:**
    Agent drift occurs when model updates, prompt changes, or shifting external tool responses cause an agent's behavior to degrade over time.
    **Mitigation strategy**:
    1. Track the Four Golden Signals in LangSmith: average trajectory loop count, tool error rate, token burn per task, and goal fulfillment.
    2. Set automated alert thresholds on step depth drift (e.g., alert if 7-day rolling average step count increases by > 20%).
    3. Run recurring daily evaluation experiments in LangSmith against a golden benchmark dataset to catch regression before users report issues.
    """)

with st.expander("Q4: Why should production agents use PostgresSaver instead of MemorySaver?"):
    st.markdown("""
    **Answer:**
    `MemorySaver` stores state in the Python process's volatile RAM:
    * If the container restarts, crashes, or is rescheduled by Kubernetes, all state, checkpoints, and active conversation threads are permanently lost.
    * In contrast, `PostgresSaver` persists serialized state snapshots in an external relational database:
      * Guarantees durable ACID transactions.
      * Enables horizontal scaling across multiple stateless agent worker pods.
      * Supports long-running workflows spanning days or weeks.
    """)

with st.expander("Q5: How do you debug an agent stuck in a repetitive tool loop using LangSmith?"):
    st.markdown("""
    **Answer:**
    1. Open the LangSmith trace and locate the repeating child spans under the agent node.
    2. Inspect the **Input vs Output state diff** across successive cycles.
    3. Identify why the model is failing to progress:
       * Did the tool return an error message that the model misinterpreted?
       * Is the model generating slightly modified queries that still return irrelevant data?
       * Is the conditional routing function failing to evaluate the termination criteria?
    4. Use LangSmith's **Fork Thread** capability to test improved prompt instructions or revised schema hints directly against the failing state snapshot.
    """)

st.markdown("### Production Traps & Failure Modes")

pa_tr_col1, pa_tr_col2 = st.columns(2)

with pa_tr_col1:
    st.error("""
    **Trap 1: Synchronous Blocking in Human Review Gates**
    * **Failure**: Using `time.sleep()` or polling loops inside a node while waiting for a human to approve an action.
    * **Consequence**: Ties up API server worker threads, exhaust connection pools, and crashes the container if the human takes more than 60 seconds to respond.
    * **Remedy**: Always use native `interrupt()` which suspends execution, persists state to disk, and frees all memory threads immediately.
    """)

    st.error("""
    **Trap 2: Runaway Token Burn in Autonomous Agent Loops**
    * **Failure**: Deploying an autonomous agent without a hard `recursion_limit` ceiling or token budget.
    * **Consequence**: An agent stuck in a reasoning loop can consume millions of tokens overnight, generating thousands of dollars in surprise cloud API bills.
    * **Remedy**: Enforce strict `recursion_limit` parameters and configure LangSmith cost alert webhooks at the project level.
    """)

with pa_tr_col2:
    st.warning("""
    **Trap 3: Unbounded State Channel Accumulation**
    * **Failure**: Allowing state channels to accumulate raw tool payloads, screenshots, and file buffers indefinitely across a multi-day thread.
    * **Consequence**: State serialization latency increases exponentially, database storage costs skyrocket, and LLM context windows overflow.
    * **Remedy**: Implement message trimming nodes and save large artifacts to S3/blob storage, referencing only pointers in the agent state.
    """)

    st.warning("""
    **Trap 4: Circular Handoff Ping-Pong in Swarm Architectures**
    * **Failure**: Agent A delegates to Agent B, who delegates back to Agent A without making task progress.
    * **Consequence**: Infinite delegation loops that consume latency budgets and frustrate users.
    * **Remedy**: Implement an explicit handoff counter in the state schema and restrict peer agents from delegating back to the same source.
    """)

st.markdown("### Core Technical Flashcards")

fcpa_c1, fcpa_c2 = st.columns(2)

with fcpa_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does `interrupt()` pause execution in LangGraph v0.2+?")
        st.markdown("**Back**: It suspends the graph super-step, serializes the current state snapshot to the checkpointer, and halts the invocation without blocking active server threads.")

    with fcpa_c2:
        st.markdown("**Front**: What are the Four Golden Signals of production agent telemetry?")
        st.markdown("**Back**: 1. Token Burn Velocity & Cost, 2. Tool Execution Success Rate, 3. Trajectory Step Depth, and 4. HITL Approval Latency.")

with fcpa_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does a client resume an interrupted LangGraph state machine?")
        st.markdown("**Back**: By invoking `app.invoke(Command(resume=payload), config=thread_config)`. The payload passed to resume becomes the return value of the `interrupt()` call.")

    with fcpa_c2:
        st.markdown("**Front**: What is the primary operational risk of monolithic agents vs multi-agent swarms?")
        st.markdown("**Back**: Monolithic agents suffer prompt bloat, tool selection confusion, and single-point reasoning failures, whereas multi-agent swarms provide modular role specialization and isolated error boundaries.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of production agent operations and multi-agent observability:")

pa_chk1 = st.checkbox("I understand how durable checkpointers (PostgresSaver) support long-running multi-day agents.", value=True)
pa_chk2 = st.checkbox("I can manage thread-scoped working memory and cross-session episodic knowledge in BaseStore.", value=True)
pa_chk3 = st.checkbox("I know how to implement deterministic human review gates using `interrupt()` and `Command(resume=...)`.", value=True)
pa_chk4 = st.checkbox("I understand the architectural trade-offs between Supervisor and Peer-to-Peer Swarm topologies.", value=True)
pa_chk5 = st.checkbox("I can monitor the Four Golden Signals of agent telemetry and set up real-time anomaly alerts.", value=True)
pa_chk6 = st.checkbox("I know how to use LangSmith trace inspection and Time-Travel forking to debug runaway agent loops.", value=True)

if pa_chk1 and pa_chk2 and pa_chk3 and pa_chk4 and pa_chk5 and pa_chk6:
    st.success("Mastery Certified: You have mastered production agent operations, multi-agent observability, and human-in-the-loop governance in LangSmith.")
