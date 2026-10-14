import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="27.1 Deep Agent Concepts - LangChain Architecture",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for deep agent cognition cards and planning tree telemetry
st.markdown("""
<style>
    .deep-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-deep-plan {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-deep-react {
        background-color: #78350F;
        color: #FDE68A;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DeepAgentCognitionSimulator:
    """
    Simulation engine showcasing differences in reasoning depth,
    planning trees, and context compression between shallow ReAct and Deep Agents.
    """
    def compare_agents(self, goal_complexity: int) -> Dict[str, Any]:
        # Shallow ReAct Agent Metrics
        react_steps = goal_complexity * 2
        react_context_bloat = 1200 + (goal_complexity * 1800)
        react_success = goal_complexity <= 2

        # Deep Agent Metrics
        deep_milestones = max(2, goal_complexity)
        deep_subagents_spawned = max(1, goal_complexity - 1)
        deep_compressed_context = 850 + (goal_complexity * 350)
        deep_success = True

        return {
            "react": {
                "architecture": "Shallow ReAct Loop (Thought -> Action -> Observation)",
                "steps": react_steps,
                "token_load": react_context_bloat,
                "success": react_success,
                "failure_reason": "Context window saturation and lost planning anchor" if not react_success else "None"
            },
            "deep": {
                "architecture": "Deep Agent (Hierarchical Planning + Subagents + Memory Paging)",
                "milestones": deep_milestones,
                "subagents": deep_subagents_spawned,
                "token_load": deep_compressed_context,
                "success": deep_success,
                "efficiency_gain": round((1 - (deep_compressed_context / max(1, react_context_bloat))) * 100, 1)
            }
        }


# --- LESSON HEADER ---
st.title("27.1 Deep Agent Concepts & Cognitive Architecture")
st.caption("Chapter 27: Deep Agents | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Traditional LLM agents operate on shallow **ReAct loops** (Reasoning + Acting): taking an action, observing the output,
and immediately deciding the next turn. When confronted with multi-hour, enterprise engineering tasks, ReAct loops
fail catastrophically due to context exhaustion and lack of foresight.

**Deep Agents** introduce a new paradigm: hierarchical planning, self-directed exploration, recursive subagent spawning,
durable state memory, and proactive context compression.

In this lesson, you will master:
1. What defines a Deep Agent and how it transcends standard ReAct loops.
2. The architectural differences between single-turn LangChain agents and Deep Agents.
3. How Deep Agents build upon and integrate with LangGraph cyclic state machines.
4. Hierarchical Planning, Tree-of-Thought decomposition, and adaptive milestone tracking.
5. Recursive Subagent Spawning for isolated problem solving.
6. Context Management, memory paging, and observation pruning.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Deep Agent Cognitive Runtime", divider="rainbow")
st.markdown("""
A Deep Agent operates as an autonomous cognitive system. It parses an open-ended goal into a milestone dependency graph,
manages working memory, delegates tasks to ephemeral subagents, and continuously audits its own progress.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    deep_dot = """
    digraph DeepAgentBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_meta_planner {
            label="1. Strategic Meta-Planner Layer";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            goal_in [label="Open-Ended Enterprise Goal\\n'Refactor auth module, add OAuth2, run tests'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            planner_engine [label="Hierarchical Planner\\nDecomposes into DAG Milestones", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            plan_dag [label="Plan Dependency Graph\\n[M1: Audit] -> [M2: Code] -> [M3: Test]", fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_execution_engine {
            label="2. Autonomous Execution & Subagents";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            exec_core [label="Executive Controller\\nMonitors milestone execution", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            sub_1 [label="Subagent A (Repo Search)\\nPrivate 15k token scratchpad", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sub_2 [label="Subagent B (Code Generator)\\nPrivate sandbox execution", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            artifact_box [label="Harvested Artifacts\\nClean diffs & test results", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_memory_reflection {
            label="3. Context Compression & Self-Reflection";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            compressor [label="Context Paging & Compression\\nPrunes verbose tool outputs, keeps facts", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            reflection_loop [label="Self-Reflection & Replanning\\nDid milestone M2 pass tests? If no, adapt", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
            durable_checkpointer [label="Durable Checkpointer\\nSnapshots state across hours/days", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        goal_in -> planner_engine [label="Objective"];
        planner_engine -> plan_dag [label="Generates Milestones"];
        plan_dag -> exec_core [label="Dispatches M1"];
        exec_core -> sub_1 [label="Spawns Isolated Subagent"];
        sub_1 -> artifact_box [label="Returns Findings"];
        artifact_box -> compressor [label="Compresses Findings"];
        compressor -> reflection_loop [label="Audits Output"];
        reflection_loop -> plan_dag [label="Updates Status / Replans"];
        reflection_loop -> durable_checkpointer [label="Persists Progress"];
    }
    """
    st.graphviz_chart(deep_dot, use_container_width=True)

# --- SECTION 2: WHAT ARE DEEP AGENTS? ---
st.header("2. What are Deep Agents?", divider="rainbow")
st.markdown("""
A **Deep Agent** is an autonomous, long-running intelligence system capable of recursive reasoning,
independent planning, tool-driven self-correction, and multi-hour execution without continuous human guidance.

While a shallow agent answers single-turn questions or runs a 3-step retrieval chain, a Deep Agent operates like an
autonomous engineering coworker: exploring an unfamiliar codebase, formulating a hypothesis, executing tests,
debugging exceptions, compressing its memory, and delivering a verified pull request.
""")

col_da1, col_da2 = st.columns(2)
with col_da1:
    st.markdown("""
    #### The Four Pillars of Deep Agents:
    1. **Strategic Foresight (Planning)**: Synthesizes a structured plan before taking actions; refuses to blindly invoke tools without a roadmap.
    2. **Recursive Decomposition (Subagents)**: Delegates difficult sub-problems to child agents with isolated token contexts.
    3. **Durable Persistence**: Checkpoints internal cognitive state to disk; can pause, sleep, recover from crashes, and resume across days.
    4. **Active Self-Correction (Reflection)**: Evaluates its own intermediate outputs against ground-truth unit tests or compilation checks, adapting its plan dynamically.
    """)

with col_da2:
    with st.container(border=True):
        st.markdown("💡 **Deep Agents vs Standard Automation**")
        st.caption("Standard scripts follow fixed if/else paths. ReAct agents stumble forward one turn at a time.")
        st.markdown("""
        Deep Agents bridge the gap between high-level human objectives and low-level execution:
        - **Human Goal**: `"Fix the memory leak in the billing service."`
        - **Deep Agent Workflow**: Analyzes git logs -> Reproduces leak with benchmark -> Localizes unclosed database cursor -> Modifies code -> Verifies benchmark -> Opens PR.
        """)

# --- SECTION 3: DEEP AGENTS VS LANGCHAIN AGENTS ---
st.header("3. Deep Agents vs LangChain Agents", divider="rainbow")
st.markdown("""
Standard LangChain agents (historically built with `AgentExecutor` or standard tool loops) operate on linear ReAct principles.
Deep Agents fundamentally overhaul the execution model.
""")

st.subheader("Architectural Comparison Matrix")
comp_agents = pd.DataFrame({
    "Dimension": ["Planning Horizon", "Execution Duration", "Context Management", "Subagent Spawning", "Error Handling Strategy", "Max Consecutive Steps"],
    "Standard LangChain ReAct Agent": ["Greedy / 1-step foresight", "10 seconds to 1 minute", "Linear append-only (Prone to saturation)", "Unsupported (Monolithic thread)", "Re-prompts model with error string", "Collapses at 8-10 turns"],
    "Deep Agent System": ["Hierarchical DAG / Multi-milestone", "Hours to multiple days", "Active compression & memory paging", "Recursive, isolated child runtimes", "Self-reflection, rollback, replanning", "Scales to hundreds of coordinated steps"]
})
st.dataframe(comp_agents, use_container_width=True)

# --- SECTION 4: DEEP AGENTS VS LANGGRAPH ---
st.header("4. Deep Agents vs LangGraph", divider="rainbow")
st.markdown("""
A common confusion is whether Deep Agents are an alternative to LangGraph. In reality, **LangGraph is the foundational
engine upon which Deep Agents are built**.
""")

col_lg1, col_lg2 = st.columns(2)
with col_lg1:
    st.markdown("""
    #### LangGraph (The Infrastructure):
    - Provides cyclic state machines, node transitions, and edges.
    - Manages checkpointing (`SqliteSaver`, `PostgresSaver`) for pause/resume.
    - Implements channel reducers (`add_messages`) and human interrupts (`interrupt()`).
    - Acts as the low-level operating system for stateful execution.
    """)

with col_lg2:
    st.markdown("""
    #### Deep Agents (The Cognitive Architecture):
    - Implemented *inside* a LangGraph state machine.
    - Enforces planning schemas, milestone dependency tracking, and goal evaluation.
    - Manages subagent process trees and observation summarization algorithms.
    - Encapsulates domain expertise (coding, research, data forensics).
    """)

# --- SECTION 5: PLANNING ---
st.header("5. Planning", divider="rainbow")
st.markdown("""
Shallow agents plunge into execution immediately. Deep Agents mandate an explicit **Pre-Flight Planning Phase**.
The agent analyzes the user's objective, queries environment constraints, and formulates a structured milestone plan.
""")

plan_dot = """
digraph PlanningLifecycle {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    init_goal [label="Goal Input", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    plan_gen [label="1. Generate Plan\\nPydantic DAG Milestones", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    milestone_exec [label="2. Execute Milestone\\nRun tools & gather evidence", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    evaluate_node [label="3. Evaluate Outcome\\nSuccess or Failure?", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    replan_node [label="4. Adaptive Replanning\\nRefine milestones", fillcolor="#7F1D1D", fontcolor="#FCA5A5", color="#EF4444"];

    init_goal -> plan_gen [label="Parses"];
    plan_gen -> milestone_exec [label="Active Milestone"];
    milestone_exec -> evaluate_node [label="Artifact"];
    evaluate_node -> milestone_exec [label="Next Milestone"];
    evaluate_node -> replan_node [label="Failed Criteria"];
    replan_node -> milestone_exec [label="Adapted Plan"];
}
"""
st.graphviz_chart(plan_dot, use_container_width=True)

st.markdown("""
#### Tree-of-Thought (ToT) & Dynamic Replanning:
If milestone 2 encounters an unexpected roadblock (e.g., an undocumented API response or failing test),
the Deep Agent does not blindly crash. It invokes its replanning node, modifies the downstream milestones,
and continues advancing toward the overarching goal.
""")

# --- SECTION 6: SUBAGENTS ---
st.header("6. Subagents", divider="rainbow")
st.markdown("""
Deep Agents utilize **Subagents** as cognitive subprocesses. When a task requires an exploratory detour
(such as scanning 200 files in a repository or executing 30 unit tests), delegating to a subagent protects the parent's
context window from catastrophic token bloat.
""")

col_sub1, col_sub2 = st.columns(2)
with col_sub1:
    st.markdown("""
    #### Subagent Execution Protocol:
    1. **Spawning**: The parent agent defines a focused child mission: `"Inspect /src/auth/ and locate token expiry logic."`
    2. **Isolation**: The child agent runs in its own sandboxed context, consuming 20,000 tokens of file search observations.
    3. **Harvesting**: The child returns a concise 150-token artifact: `"Token expiry is handled in auth.py:L142 using variable TOKEN_TTL."`
    4. **Garbage Collection**: The child's 20,000-token scratchpad is discarded. The parent continues with its context clean and focused.
    """)

with col_sub2:
    with st.container(border=True):
        st.markdown("📦 **Subagent Spawning in Python**")
        st.code("""
def spawn_research_subagent(target_query: str) -> str:
    # 1. Independent ephemeral state
    subagent_state = {"messages": [HumanMessage(content=target_query)]}
    
    # 2. Run dedicated subagent graph
    subagent_runner = create_subagent_runner(tools=[grep_search, read_file])
    final_child_state = subagent_runner.invoke(subagent_state)
    
    # 3. Return only the synthesized artifact
    return final_child_state["messages"][-1].content
        """, language="python")

# --- SECTION 7: CONTEXT MANAGEMENT ---
st.header("7. Context Management", divider="rainbow")
st.markdown("""
Long-running agents inevitably generate hundreds of thousands of tokens of observations, shell traces, and tool results.
Without active **Context Management**, an agent will hit model context limits or experience severe attention degradation ('needle-in-a-haystack' failure).
""")

st.subheader("Deep Agent Context Management Strategies")
ctx_c1, ctx_c2, ctx_c3 = st.columns(3)
with ctx_c1:
    with st.container(border=True):
        st.markdown("✂️ **Observation Truncation**")
        st.caption("Immediate Ingress Pruning")
        st.markdown("Caps large tool returns (e.g., truncate shell outputs to first 50 lines and last 20 lines with a summary hash).")

with ctx_c2:
    with st.container(border=True):
        st.markdown("📑 **Memory Paging & Scratchpads**")
        st.caption("Active State Summarization")
        st.markdown("Periodically compresses older conversation turns into a structured 'Working Memory Scratchpad', replacing raw message history.")

with ctx_c3:
    with st.container(border=True):
        st.markdown("🗄️ **External Artifact Offloading**")
        st.caption("Disk-Backed State")
        st.markdown("Writes large files and dataset schemas to disk; passes only file paths and hashes inside prompt context.")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Autonomous Deep Agent Controller", divider="rainbow")
st.markdown("""
Below is a runnable implementation demonstrating a Deep Agent controller with structured milestone planning,
milestone progress auditing, and observation summarization.
""")

prod_deep_code = '''from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage

# 1. Structured Milestone Plan Schema
class Milestone(BaseModel):
    id: int
    title: str
    status: Literal["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"] = "PENDING"
    verification_criteria: str

class DeepAgentPlan(BaseModel):
    objective: str
    milestones: List[Milestone]

# 2. Deep Agent Controller Engine
class DeepAgentController:
    def __init__(self, objective: str):
        self.objective = objective
        self.plan = self._generate_initial_plan()
        self.scratchpad: List[str] = []

    def _generate_initial_plan(self) -> DeepAgentPlan:
        # In production: ChatModel.with_structured_output(DeepAgentPlan)
        return DeepAgentPlan(
            objective=self.objective,
            milestones=[
                Milestone(id=1, title="Audit Repository Codebase", verification_criteria="Identify vulnerable dependencies"),
                Milestone(id=2, title="Refactor Vulnerable Imports", verification_criteria="Update to secure packages"),
                Milestone(id=3, title="Run Test Suite & Verify Build", verification_criteria="100% unit tests pass")
            ]
        )

    def execute_next_milestone(self) -> dict:
        for m in self.plan.milestones:
            if m.status == "PENDING":
                m.status = "IN_PROGRESS"
                # Execute milestone via tools / subagent
                artifact = f"Verified: Successfully completed milestone {m.id} ({m.title})."
                self.scratchpad.append(artifact)
                m.status = "COMPLETED"
                return {"milestone_id": m.id, "status": "COMPLETED", "artifact": artifact}
                
        return {"status": "MISSION_COMPLETE"}

    def get_compressed_context(self) -> str:
        """Returns compressed working memory rather than raw tokens."""
        completed = [f"M{m.id} ({m.title}): COMPLETED" for m in self.plan.milestones if m.status == 'COMPLETED']
        return f"Objective: {self.objective}\\nProgress: {'; '.join(completed)}"
'''
st.code(prod_deep_code, language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Deep Agent vs ReAct Agent Reasoning Depth Simulator", divider="rainbow")
st.markdown("""
Compare how a standard ReAct agent and a Deep Agent perform across missions of escalating complexity.
Observe how context compression and hierarchical subagents prevent catastrophic failure on deep tasks.
""")

deep_engine = DeepAgentCognitionSimulator()

col_dw1, col_dw2 = st.columns([1, 1])

with col_dw1:
    st.subheader("⚙️ Mission Complexity & Reasoning Depth")
    mission_difficulty = st.slider("Select Mission Difficulty:", min_value=1, max_value=5, value=3)
    st.caption("1 = Quick API call | 3 = Multi-file Bug Fix | 5 = Autonomous Full-Stack Migration")
    run_comp_btn = st.button("🚀 Evaluate Reasoning Depth & Context Load", type="primary", use_container_width=True)

with col_dw2:
    st.subheader("📊 Cognitive Architecture Comparison")
    if run_comp_btn:
        res = deep_engine.compare_agents(mission_difficulty)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### ⚡ Standard ReAct Agent")
            st.metric("Total Ingested Tokens", f"{res['react']['token_load']:,}")
            st.metric("Total Reasoning Turns", res['react']['steps'])
            if res['react']['success']:
                st.success("Status: COMPLETED")
            else:
                st.error(f"Status: FAILED ({res['react']['failure_reason']})")

        with col_c2:
            st.markdown("##### 🧠 Deep Agent Architecture")
            st.metric("Compressed Token Load", f"{res['deep']['token_load']:,}", delta=f"-{res['deep']['efficiency_gain']}% tokens")
            st.metric("Subagents Spawned", res['deep']['subagents'])
            st.success("Status: COMPLETED (Verified via Milestones)")

        st.divider()
        st.markdown("##### 🔍 Deep Agent Strategic Advantage")
        if mission_difficulty >= 3:
            st.error("⚠️ **Shallow ReAct Failure**: At difficulty >= 3, the ReAct agent collapses because raw tool outputs saturate the context window, causing the model to forget its initial goal.")
            st.success("✅ **Deep Agent Resilience**: The Deep Agent maintains stability by breaking tasks into milestones and delegating heavy observations to ephemeral subagents.")
        else:
            st.info("💡 **Observation**: For simple difficulty 1 tasks, standard ReAct is sufficient. Deep Agent architecture shines on multi-step enterprise workflows.")
    else:
        st.info("Adjust difficulty and click 'Evaluate Reasoning Depth & Context Load'.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: What is the fundamental difference between a ReAct agent and a Deep Agent?"):
    st.markdown("""
    **Answer**: ReAct agents operate on a greedy, single-step horizon: they observe the last tool output and immediately choose
    the next action without maintaining a macro-level plan. They suffer from context window saturation and drift within 5 to 10 turns.
    Deep Agents employ hierarchical pre-flight planning, maintain milestone dependency graphs, spawn ephemeral subagents for isolated
    exploration, and actively compress their working memory, allowing them to solve multi-hour tasks requiring hundreds of steps.
    """)

with st.expander("Q2: Why is Subagent Spawning critical for long-running Deep Agents?"):
    st.markdown("""
    **Answer**: To prevent context window pollution and attention degradation. If an agent needs to search through 100 files,
    performing that search in the primary context window injects tens of thousands of tokens of noisy file contents.
    By spawning an ephemeral subagent, the exploratory tokens are confined to the child's sandbox. The child returns only
    a 2-line summary to the parent, keeping the parent's memory focused and token-efficient.
    """)

with st.expander("Q3: How do Deep Agents handle unexpected tool errors or test failures during a milestone?"):
    st.markdown("""
    **Answer**: Unlike simple scripts that crash or ReAct agents that repeatedly re-run the same failing command, Deep Agents
    incorporate a **Reflection and Replanning Node**. Upon detecting a failed verification criterion, the agent pauses,
    analyzes the error diagnostics, updates its internal milestone DAG (inserting a debugging subtask or choosing an alternative strategy),
    and resumes execution adaptively.
    """)

with st.expander("Q4: What is Context Compression in Deep Agent runtimes?"):
    st.markdown("""
    **Answer**: Context compression encompasses techniques that distill conversational memory into high-density state representations.
    This includes: (1) Truncating verbose shell and grep outputs; (2) Periodic LLM summarization of past turns into a concise
    'Working Memory Scratchpad'; and (3) Offloading large data payloads to disk and maintaining only hashes and file paths in the prompt.
    """)

with st.expander("Q5: What role does durable checkpointing play in Deep Agent architectures?"):
    st.markdown("""
    **Answer**: Because Deep Agents can run for hours or across multiple days, they cannot reside solely in volatile process RAM.
    Durable checkpointers (persisting state snapshots into Postgres or SQLite on every turn) allow the agent to survive server restarts,
    survive network drops, pause for human approval, and resume from the exact last verified milestone without restarting from scratch.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Planning Paralysis Loop)**: An agent that spends 10 sequential turns continuously refining and re-evaluating its plan without ever executing a concrete tool action. Enforce a maximum planning turn ceiling.")
    st.warning("**Trap 2 (Uncompressed Scratchpad Bloat)**: Appending raw terminal command outputs directly to the parent context. A single 10,000-line stack trace can instantly exhaust the context window and blow your token budget.")
with col_trap2:
    st.error("**Trap 3 (Unbounded Subagent Trees)**: Allowing subagents to recursively spawn further subagents without depth limits, causing an exponential process explosion that exhausts server threads and API rate limits.")
    st.warning("**Trap 4 (Omitting Deterministic Verification)**: Relying solely on the LLM's opinion ('I think this code is correct') to mark a milestone complete rather than running an actual unit test or compiler check.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What defines a Deep Agent?")
        st.caption("Flip for answer:")
        st.info("**Back**: An autonomous system capable of hierarchical planning, subagent spawning, context compression, and multi-hour self-directed problem solving.")
    with st.container(border=True):
        st.markdown("**Front**: Why do shallow ReAct loops fail on enterprise tasks?")
        st.caption("Flip for answer:")
        st.info("**Back**: They lack macro-planning, cannot isolate context, and saturate memory within 5-10 turns, losing their primary objective.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of an Ephemeral Subagent?")
        st.caption("Flip for answer:")
        st.info("**Back**: To solve an exploratory subtask in an isolated token sandbox, returning only the final artifact and discarding the noisy scratchpad.")
    with st.container(border=True):
        st.markdown("**Front**: How does LangGraph support Deep Agents?")
        st.caption("Flip for answer:")
        st.info("**Back**: It provides the stateful cyclic state machine, checkpointer persistence, and human interrupt mechanics that make deep execution durable.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand why Deep Agents outperform shallow ReAct agents on long-running tasks.", value=True)
c2 = st.checkbox("I know how to structure hierarchical milestone plans using Pydantic.", value=True)
c3 = st.checkbox("I understand the architectural relationship between Deep Agents and LangGraph.", value=True)
c4 = st.checkbox("I know how subagent spawning prevents context window saturation.", value=True)
c5 = st.checkbox("I understand context compression, memory paging, and observation pruning strategies.", value=True)
c6 = st.checkbox("I recognize how durable checkpointers enable multi-hour crash-resilient execution.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 27.2: Deep Agent Capabilities.")
