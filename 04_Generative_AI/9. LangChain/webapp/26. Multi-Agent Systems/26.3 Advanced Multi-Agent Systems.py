import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="26.3 Advanced Multi-Agent Systems - LangChain Architecture",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mesh telemetry and state cards
st.markdown("""
<style>
    .mesh-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-mesh-status {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-mesh-deadlock {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MultiAgentMeshSimulator:
    """
    Simulation engine showcasing dynamic agent handoffs, shared state synchronization,
    deadlock detection, and self-healing failure recovery.
    """
    def __init__(self):
        self.call_history = []

    def simulate_mesh_execution(self, trigger_failure: bool, max_hops: int = 5) -> Dict[str, Any]:
        trace = []
        current_agent = "ingress_triage"
        state = {"task_id": "T-8891", "artifacts": [], "iteration": 0}
        deadlock_detected = False
        recovered = False

        transitions = {
            "ingress_triage": "researcher",
            "researcher": "coder",
            "coder": "tester",
            "tester": "FINISH"
        }

        for hop in range(max_hops):
            state["iteration"] += 1
            trace.append({"hop": hop + 1, "agent": current_agent, "action": f"Executed node logic for {current_agent}"})

            if trigger_failure and current_agent == "coder" and hop == 2:
                # Simulate worker deadlock / failure
                trace.append({"hop": hop + 1, "agent": "coder", "action": "ERROR: Coder raised APIConnectionTimeout!"})
                trace.append({"hop": hop + 1, "agent": "CIRCUIT_BREAKER", "action": "Interception: Rerouted to fallback_coder node"})
                current_agent = "fallback_coder"
                recovered = True
                continue

            if current_agent == "fallback_coder":
                current_agent = "tester"
                continue

            next_agent = transitions.get(current_agent, "FINISH")
            if next_agent == "FINISH":
                trace.append({"hop": hop + 2, "agent": "SYSTEM", "action": "Mission Finalized: All state channels verified."})
                break
            current_agent = next_agent

        return {
            "trace": trace,
            "recovered": recovered,
            "total_hops": len(trace),
            "final_status": "SUCCESS" if (current_agent in ["tester", "FINISH"] or recovered) else "DEADLOCK"
        }


# --- LESSON HEADER ---
st.title("26.3 Advanced Multi-Agent Systems & Resilient Meshes")
st.caption("Chapter 26: Multi-Agent Systems | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
At enterprise production scale, multi-agent systems evolve beyond static supervisor scripts into resilient,
self-healing distributed meshes. These architectures feature dynamic agent handoffs, distributed shared memory,
ephemeral subagent spawning, race-condition mitigation, and comprehensive trajectory evaluation.

In this lesson, you will master:
1. Orchestrating explicit **Agent Handoffs** with context-preserving control transfers.
2. Managing **Shared State** across typed schemas and concurrent channel reducers.
3. Architecting **Agent Memory** spanning short-term thread scratchpads and long-term episodic storage.
4. Implementing bounded **Agent Delegation** and permission inheritance.
5. Spawning isolated, ephemeral **Subagents** for parallel sub-problem execution.
6. Safeguarding **Parallel Task Execution** against race conditions and state clobbering.
7. Engineering **Agent Failure Recovery** via circuit breakers, loop detectors, and self-healing nodes.
8. Benchmarking multi-agent trajectory quality and token efficiency.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Advanced Distributed Multi-Agent Mesh", divider="rainbow")
st.markdown("""
An advanced multi-agent mesh connects specialized cognitive agents through a shared state fabric,
a persistent checkpointer, dynamic handoff tools, and a supervisor supervisor-watchdog engine.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    mesh_dot = """
    digraph AdvancedMeshBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_state_fabric {
            label="1. Global State Fabric & Reducers";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            global_state [label="Typed State Schema\\nmessages: Annotated[list, add_messages]\\nartifacts: dict\\nactive_agent: str", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            checkpointer [label="Checkpointer (Sqlite / Postgres)\\nPersists thread snapshots per turn", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_agent_mesh {
            label="2. Dynamic Agent Mesh (Handoff Nodes)";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            triage_agent [label="Triage Agent\\nInspects intent, performs initial handoff", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            coder_agent [label="Coder Agent\\nSpawns test subagent", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            sub_tester [label="Ephemeral Subagent\\nExecutes unit test sandbox", fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            auditor_agent [label="Auditor Agent\\nVerifies compliance", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_watchdog {
            label="3. Resilience & Failure Recovery";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            deadlock_monitor [label="Deadlock Watchdog\\nRecursion ceiling & loop detector", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            fallback_node [label="Circuit Breaker / Fallback\\nReroutes to human or deterministic path", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        global_state -> triage_agent [label="Hydrates Context"];
        triage_agent -> coder_agent [label="transfer_to_coder()"];
        coder_agent -> sub_tester [label="Spawns Child"];
        sub_tester -> coder_agent [label="Returns Test Logs"];
        coder_agent -> auditor_agent [label="transfer_to_auditor()"];
        auditor_agent -> global_state [label="Writes Final Artifact"];
        coder_agent -> deadlock_monitor [label="Telemetry Stream"];
        deadlock_monitor -> fallback_node [label="On Exception / Loop"];
    }
    """
    st.graphviz_chart(mesh_dot, use_container_width=True)

# --- SECTION 2: AGENT HANDOFFS ---
st.header("2. Agent Handoffs", divider="rainbow")
st.markdown("""
An **Agent Handoff** is an explicit transfer of execution control from one agent to another.
In modern LangGraph and multi-agent systems, handoffs are implemented using **Handoff Tools** or `Command(goto=...)` objects.
""")

col_ho1, col_ho2 = st.columns(2)
with col_ho1:
    st.markdown("""
    #### Anatomy of a Handoff Tool
    When Agent A determines that a task falls outside its specialization, it calls a dedicated tool:
    - Tool Name: `transfer_to_billing(reason=...)`
    - Function: Emits an instruction message and returns a routing directive to the state machine.
    - Context Isolation: The tool passes an executive summary to the incoming agent, avoiding raw context dumps.
    """)

with col_ho2:
    with st.container(border=True):
        st.markdown("🔄 **LangGraph Command Handoff Pattern**")
        st.code("""
from langgraph.types import Command
from langchain_core.tools import tool

@tool
def transfer_to_coder(task_brief: str) -> Command:
    \"\"\"Hands control over to the Software Engineering specialist.\"\"\"
    return Command(
        goto="coder_node",
        update={
            "messages": [AIMessage(content=f"Transferred to coder: {task_brief}", name="triage")],
            "active_agent": "coder"
        }
    )
        """, language="python")

# --- SECTION 3: SHARED STATE ---
st.header("3. Shared State", divider="rainbow")
st.markdown("""
In a multi-agent system, agents do not maintain completely isolated universes. They share a common blackboard
known as the **State Schema**. Managing concurrent updates without data clobbering requires **Channel Reducers**.
""")

st.subheader("State Channel Reducers in Python TypedDict")
st.code("""
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph.message import add_messages

class MultiAgentGlobalState(TypedDict):
    # Appends new messages from any agent without overwriting existing history
    messages: Annotated[List[Any], add_messages]
    
    # Custom reducer merging dictionary keys across parallel workers
    shared_artifacts: Annotated[Dict[str, Any], lambda a, b: {**a, **b}]
    
    # Current active worker token
    active_worker: str
    
    # Global iteration counter
    iteration_count: int
""", language="python")

# --- SECTION 4: AGENT MEMORY ---
st.header("4. Agent Memory", divider="rainbow")
st.markdown("""
Multi-agent memory operates across two distinct temporal planes:
**Short-Term Working Memory** (the active execution thread) and **Long-Term Episodic Memory** (cross-thread corporate knowledge).
""")

mem_col1, mem_col2 = st.columns(2)
with mem_col1:
    with st.container(border=True):
        st.markdown("🧠 **Short-Term Thread Memory**")
        st.caption("Scope: Single Execution Session")
        st.markdown("""
        - Stored in checkpointer memory (`MemorySaver`, `SqliteSaver`).
        - Tracks recent tool calls, intermediate thoughts, and active handoffs.
        - Automatically pruned or summarized when context exceeds 80% capacity.
        """)

with mem_col2:
    with st.container(border=True):
        st.markdown("💾 **Long-Term Episodic Memory**")
        st.caption("Scope: Cross-Session & Historical Knowledge")
        st.markdown("""
        - Backed by an external vector database (Pinecone, Chroma) or key-value store.
        - Stores past project debriefs, learned user preferences, and error corrections.
        - Queried via semantic similarity during ingress initialization.
        """)

# --- SECTION 5: AGENT DELEGATION ---
st.header("5. Agent Delegation", divider="rainbow")
st.markdown("""
**Delegation** allows an agent to subcontract a subtask to another agent while retaining ultimate responsibility for the final outcome.
Unlike an irreversible handoff (where Agent A permanently hands over control), delegation acts like a remote procedure call:
Agent A invokes Agent B, waits for Agent B's result, and resumes execution.
""")

st.markdown("""
#### Critical Safeguards for Agent Delegation:
- **Maximum Delegation Depth**: Hardcode a recursion ceiling (e.g., maximum call depth of 3) to prevent runaway chain reactions.
- **Permission Boundary Inheritance**: Subcontracted agents cannot hold broader tool permissions than the delegating parent.
- **Time-to-Live (TTL) Budgets**: Each delegated subtask must have a strict timeout (e.g. 10 seconds).
""")

# --- SECTION 6: SUBAGENTS ---
st.header("6. Subagents", divider="rainbow")
st.markdown("""
**Subagents** are ephemeral child agents spawned to solve an isolated sub-problem. Once their specific task is solved,
the subagent's execution environment and scratchpad are garbage collected, leaving only the clean final artifact in the parent's context.
""")

sub_dot = """
digraph SubagentSpawning {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    parent_agent [label="Parent Coder Agent\\nTask: Build authentication system", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    child_subagent [label="Ephemeral Child Subagent\\nIsolated Task: Run 50 unit tests & fix bugs", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    clean_artifact [label="Clean Summary Returned\\n'All 50 tests passed in 4 iterations.'", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

    parent_agent -> child_subagent [label="Spawns Child Subagent"];
    child_subagent -> child_subagent [label="Iterates in private scratchpad (20k tokens)"];
    child_subagent -> clean_artifact [label="Extracts verified artifact"];
    clean_artifact -> parent_agent [label="Zero token scratchpad pollution!"];
}
"""
st.graphviz_chart(sub_dot, use_container_width=True)

# --- SECTION 7: PARALLEL TASK EXECUTION ---
st.header("7. Parallel Task Execution", divider="rainbow")
st.markdown("""
When multiple agents operate concurrently on independent subtasks, applications must prevent race conditions
and handle partial failure gracefully.
""")

st.subheader("Concurrency Management Matrix")
concurrency_df = pd.DataFrame({
    "Challenge": ["Race Conditions on State", "Partial Worker Failure", "Wall-Clock Stragglers", "Token Rate Limit Spike"],
    "Consequence": ["Worker A overwrites Worker B's dictionary update", "One failed worker blocks the entire scatter-gather join", "Slowest worker dictates overall response time", "Simultaneous parallel calls trigger API 429 errors"],
    "Production Defense": ["Use Channel Reducers (`add_messages`, dict merge reducers)", "Implement `asyncio.gather(..., return_exceptions=True)` with fallbacks", "Set per-worker timeout ceilings (e.g. 5s per worker)", "Enforce global token bucket rate-limiters on async workers"]
})
st.dataframe(concurrency_df, use_container_width=True)

# --- SECTION 8: AGENT FAILURE RECOVERY ---
st.header("8. Agent Failure Recovery", divider="rainbow")
st.markdown("""
Distributed multi-agent systems will inevitably experience failures: worker timeouts, tool exceptions, hallucinated arguments,
and deadlocked loops. A resilient architecture embeds self-healing mechanics.
""")

col_rec1, col_rec2 = st.columns(2)
with col_rec1:
    st.markdown("""
    #### Self-Healing Architectural Components:
    1. **Deadlock Loop Breakers**: Detects if two agents have exchanged more than 4 consecutive messages without producing a new artifact. Automatically forces a break to a supervisor or human.
    2. **Circuit Breakers**: If a specific tool or worker agent fails 3 times consecutively, the circuit opens, rerouting tasks to a degraded heuristic fallback.
    3. **Deterministic Fallback Nodes**: When an LLM fails schema parsing, control transitions to a deterministic Python rule node rather than crashing the pipeline.
    """)

with col_rec2:
    with st.container(border=True):
        st.markdown("🛡️ **Deadlock Detection Guard**")
        st.code("""
def check_deadlock_condition(state: dict) -> bool:
    history = state.get("messages", [])
    if len(history) < 6:
        return False
    recent_senders = [m.name for m in history[-6:]]
    # Check if alternating between only two agents
    if set(recent_senders) == {"agent_a", "agent_b"}:
        return True # Deadlock detected!
    return False
        """, language="python")

# --- SECTION 9: MULTI-AGENT EVALUATION ---
st.header("9. Multi-Agent Evaluation", divider="rainbow")
st.markdown("""
Evaluating a multi-agent system requires assessing both the **Final Output Quality** and the **Trajectory Quality**
(how efficiently and safely the agents collaborated).
""")

st.subheader("Key Multi-Agent Telemetry Metrics")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    with st.container(border=True):
        st.markdown("🎯 **Trajectory Efficiency**")
        st.caption("Ratio: Optimal Steps vs Actual Steps")
        st.markdown("Measures whether agents solved the goal in 4 direct hops or wandered aimlessly through 18 redundant exchanges.")

with m_col2:
    with st.container(border=True):
        st.markdown("💰 **Token Overhead Ratio**")
        st.caption("Handoff Tokens vs Payload Tokens")
        st.markdown("Tracks what percentage of overall token consumption was spent on inter-agent pleasantries and coordination overhead.")

with m_col3:
    with st.container(border=True):
        st.markdown("🔄 **Handoff Accuracy**")
        st.caption("Correct Next Agent Selection")
        st.markdown("Measures whether the triage or supervisor agent selected the optimal specialist on the first attempt.")

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Resilient Mesh with Handoffs & Fallbacks", divider="rainbow")
st.markdown("""
Below is a runnable distributed mesh architecture with explicit handoff mechanics,
a shared state reducer, and an automatic circuit-breaker failure fallback.
""")

prod_mesh_code = '''from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# 1. Shared State Definition with Add-Messages Reducer
class MeshState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_agent: str
    iteration: int

# 2. Specialized Mesh Nodes
def triage_node(state: MeshState) -> dict:
    return {
        "messages": [AIMessage(content="Triage complete. Routing to Coder.", name="triage")],
        "active_agent": "coder",
        "iteration": state.get("iteration", 0) + 1
    }

def coder_node(state: MeshState) -> dict:
    # Simulates error handling and handoff
    return {
        "messages": [AIMessage(content="Code generated. Ready for audit.", name="coder")],
        "active_agent": "auditor",
        "iteration": state["iteration"] + 1
    }

def auditor_node(state: MeshState) -> dict:
    return {
        "messages": [AIMessage(content="Security audit passed. Mission Complete.", name="auditor")],
        "active_agent": "FINISH",
        "iteration": state["iteration"] + 1
    }

def fallback_recovery_node(state: MeshState) -> dict:
    """Circuit breaker node handling unexpected worker failures."""
    return {
        "messages": [AIMessage(content="Circuit Breaker: Degrading to safe fallback report.", name="fallback")],
        "active_agent": "FINISH",
        "iteration": state["iteration"] + 1
    }

# Dynamic Router Logic
def mesh_router(state: MeshState) -> str:
    if state["iteration"] > 8:
        return "fallback" # Enforce hard safety ceiling
    return state["active_agent"]
'''
st.code(prod_mesh_code, language="python")

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Multi-Agent Handoff & Deadlock Recovery Simulator", divider="rainbow")
st.markdown("""
Simulate an active multi-agent mesh running across specialized nodes.
Trigger synthetic worker failures to observe how the mesh detects exceptions and self-heals via circuit-breaker recovery.
""")

mesh_engine = MultiAgentMeshSimulator()

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Mesh Configuration & Fault Injection")
    trigger_fault = st.checkbox("Inject Synthetic Coder Node Failure (Simulate API Crash)", value=False)
    max_hops_slider = st.slider("Max Permitted Handoff Hops:", min_value=3, max_value=8, value=5)
    run_mesh_btn = st.button("🚀 Execute Multi-Agent Mesh Run", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Execution Trace & Self-Healing Telemetry")
    if run_mesh_btn:
        result = mesh_engine.simulate_mesh_execution(trigger_fault, max_hops_slider)

        if result["recovered"]:
            st.markdown(
                f"<div class='mesh-card'><span class='badge-mesh-deadlock'>CIRCUIT BREAKER TRIGGERED</span>"
                f"<br><br><b>Fault Injected:</b> Coder node timeout"
                f"<br><b>Self-Healing:</b> Automatically rerouted to fallback node.</div>",
                unsafe_allow_html=True
            )
            st.warning("⚠️ Worker failure intercepted. Pipeline self-healed and completed successfully.")
        else:
            st.markdown(
                f"<div class='mesh-card'><span class='badge-mesh-status'>NOMINAL MESH EXECUTION</span>"
                f"<br><br><b>Trace Status:</b> All specialist handoffs verified cleanly.</div>",
                unsafe_allow_html=True
            )
            st.success("✅ Nominal Execution: All nodes executed without deadlock.")

        st.markdown("##### 📜 Chronological Node Execution Trace")
        trace_df = pd.DataFrame(result["trace"])
        st.dataframe(trace_df[["hop", "agent", "action"]], use_container_width=True)
    else:
        st.info("Configure fault settings and click 'Execute Multi-Agent Mesh Run' to view trace telemetry.")

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: What is the architectural purpose of LangGraph Channel Reducers in Multi-Agent Systems?"):
    st.markdown("""
    **Answer**: In concurrent or sequential multi-agent execution, multiple nodes write updates back to the shared state graph.
    Without reducers, the last node to execute completely overwrites the state dictionary. A channel reducer (such as `add_messages`
    or custom list appenders) defines a deterministic mathematical merge rule, ensuring that messages from diverse agents are cleanly
    appended rather than destroyed.
    """)

with st.expander("Q2: How do ephemeral subagents prevent context window degradation in parent agents?"):
    st.markdown("""
    **Answer**: When a parent agent tackles a sub-problem that requires 20 tool iterations (e.g., iteratively compiling code,
    inspecting errors, and re-running unit tests), executing those turns in the parent context injects 40,000 redundant tokens into the parent's memory.
    A subagent runs those 20 iterations inside its own isolated context sandbox, achieves success, and returns only a 100-token summary
    artifact to the parent. The parent's context remains lean and pristine.
    """)

with st.expander("Q3: How do you detect and recover from deadlocks in decentralized multi-agent mesh networks?"):
    st.markdown("""
    **Answer**: Deadlocks occur when two or more agents cycle indefinitely in an unresolved loop (e.g., Agent A asks Agent B for data,
    Agent B asks Agent A for clarification). Recovery strategies include: (1) Graph-level recursion limits; (2) History state analyzers
    that flag when the same sequence of agent handoffs repeats 3+ times; and (3) Circuit-breaker fallback nodes that intercept
    repeating cycles and force a break to an executive supervisor or human reviewer.
    """)

with st.expander("Q4: What is the difference between an Agent Handoff and an Agent Delegation?"):
    st.markdown("""
    **Answer**: An **Agent Handoff** is an irreversible transfer of control: Agent A transfers the active execution token to Agent B,
    and Agent A is terminated from that branch of the conversation. An **Agent Delegation** is an invocation: Agent A pauses, calls Agent B
    as a helper tool, receives Agent B's return payload, and Agent A continues driving the primary execution.
    """)

with st.expander("Q5: What metrics should be tracked in LangSmith to evaluate multi-agent system efficiency?"):
    st.markdown("""
    **Answer**: Key metrics include: (1) **Trajectory Step Count**: Ratio of actual steps vs theoretical minimum;
    (2) **Coordination Overhead**: Tokens consumed by routing/handoff decisions vs actual user value delivery;
    (3) **Tool Error Attribution**: Pinpointing which specific specialist agent generates tool failures; and
    (4) **Handoff Latency**: Time elapsed during state serialization between nodes.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (State Clobbering in Multi-Branch Merges)**: Parallel agents returning dictionaries with the same keys without a designated reducer causes race-condition silent data loss in production graphs.")
    st.warning("**Trap 2 (Unbounded Subagent Nesting)**: Allowing subagents to spawn subagents infinitely creates an uncontrolled tree of execution that rapidly exhausts API rate limits and crashes server memory.")
with col_trap2:
    st.error("**Trap 3 (Passing Raw Chat Logs into Subagents)**: Feeding the entire parent history into child subagents defeats the purpose of context isolation and spikes token costs.")
    st.warning("**Trap 4 (Silent Failures in Background Workers)**: Running background parallel workers without `return_exceptions=True` causes unhandled exceptions in one worker to crash the entire application.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is a Channel Reducer in LangGraph?")
        st.caption("Flip for answer:")
        st.info("**Back**: A function defining how concurrent or sequential node updates are merged into a shared state key (e.g., `add_messages` appending chat history).")
    with st.container(border=True):
        st.markdown("**Front**: What is an Ephemeral Subagent?")
        st.caption("Flip for answer:")
        st.info("**Back**: A temporary child agent created to execute a focused multi-step subtask whose large scratchpad is discarded after returning a clean final artifact.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does a Circuit Breaker node protect multi-agent pipelines?")
        st.caption("Flip for answer:")
        st.info("**Back**: It intercepts repeating worker errors or deadlock loops and forces execution down a deterministic, graceful fallback path.")
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between working memory and episodic memory in MAS?")
        st.caption("Flip for answer:")
        st.info("**Back**: Working memory is short-term active session state (checkpointer); episodic memory is long-term persistent knowledge retrieved across past sessions (vector DB).")

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand how to implement explicit agent handoffs using Command objects.", value=True)
c2 = st.checkbox("I know how to configure shared state schemas with channel reducers in LangGraph.", value=True)
c3 = st.checkbox("I understand the distinction between ephemeral subagents and permanent worker nodes.", value=True)
c4 = st.checkbox("I know how to safeguard parallel task execution against race conditions.", value=True)
c5 = st.checkbox("I can implement deadlock loop detection and circuit-breaker failure recovery.", value=True)
c6 = st.checkbox("I understand how to evaluate multi-agent trajectories and token efficiency in LangSmith.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 26: Multi-Agent Systems.")
