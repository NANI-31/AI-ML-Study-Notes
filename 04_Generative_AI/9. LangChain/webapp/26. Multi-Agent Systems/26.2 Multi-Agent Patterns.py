import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="26.2 Multi-Agent Patterns - LangChain Architecture",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for topology patterns and agent state cards
st.markdown("""
<style>
    .pattern-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pattern-active {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MultiAgentPatternEngine:
    """
    Simulation engine evaluating execution trace dynamics across
    Supervisor, Sequential, and Parallel multi-agent patterns.
    """
    def simulate_pattern(self, pattern_type: str, num_tasks: int) -> Dict[str, Any]:
        if pattern_type == "Supervisor":
            total_llm_calls = 1 + (num_tasks * 2) # 1 ingress supervisor + (1 worker + 1 supervisor return) * tasks
            parallel_possible = False
            latency_factor = total_llm_calls * 1.1
            description = "Supervisor iteratively delegates to workers and inspects results before terminating."
        elif pattern_type == "Sequential":
            total_llm_calls = num_tasks # Linear pipeline
            parallel_possible = False
            latency_factor = num_tasks * 0.95
            description = "Assembly line: Agent 1 -> Agent 2 -> Agent 3 in strict order."
        elif pattern_type == "Parallel (Scatter-Gather)":
            total_llm_calls = num_tasks + 1 # N workers executed concurrently + 1 aggregator
            parallel_possible = True
            latency_factor = 1.0 + 1.2 # Parallel execution time is max single worker + aggregator
            description = "Concurrent execution: Workers run simultaneously, aggregator synthesizes results."
        else: # Hierarchical
            total_llm_calls = 2 + (num_tasks * 2)
            parallel_possible = True
            latency_factor = total_llm_calls * 0.8
            description = "Multi-tier leadership: Director delegates to Team Lead, who manages granular workers."

        return {
            "pattern": pattern_type,
            "total_calls": total_llm_calls,
            "latency_index_seconds": round(latency_factor, 2),
            "supports_concurrency": parallel_possible,
            "description": description
        }


# --- LESSON HEADER ---
st.title("26.2 Multi-Agent Architectural Patterns")
st.caption("Chapter 26: Multi-Agent Systems | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Designing effective multi-agent systems requires selecting the right structural design pattern.
Depending on your operational requirements—whether you need strict hierarchical oversight, high-throughput parallel execution,
or deterministic pipeline assembly—different coordination topologies offer distinct performance and governance guarantees.

In this lesson, you will master:
1. The **Supervisor Pattern** for centralized command, iterative delegation, and termination control.
2. The **Router Pattern** for high-efficiency, one-shot deterministic task classification.
3. Managing **Worker Agents** with bounded responsibilities and clean state returns.
4. Designing **Specialist Agents** with dedicated toolboxes and persona constraints.
5. **Sequential Pipelines** for staged assembly-line workflows.
6. **Parallel Scatter-Gather** patterns for concurrent execution and map-reduce synthesis.
7. **Hierarchical Topologies** for multi-tier enterprise decision architectures.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Multi-Agent Patterns Taxonomy", divider="rainbow")
st.markdown("""
Multi-agent topologies fall into four primary structural paradigms: Centralized Supervisor, Linear Sequential,
Concurrent Scatter-Gather, and Deep Hierarchical trees.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    patterns_dot = """
    digraph MultiAgentPatternsTaxonomy {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_supervisor_pattern {
            label="Pattern A: Centralized Supervisor";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            sup_core [label="Supervisor Agent\\nRoutes & Evaluates Loop", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            w_code [label="Coder Worker", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            w_docs [label="Doc Worker", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sup_core -> w_code [label="Delegates"];
            w_code -> sup_core [label="Returns"];
            sup_core -> w_docs [label="Delegates"];
            w_docs -> sup_core [label="Returns"];
        }

        subgraph cluster_sequential_pattern {
            label="Pattern B: Sequential Assembly Line";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            s_draft [label="Drafting Agent", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            s_edit [label="Editing Agent", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            s_audit [label="Security Audit Agent", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            s_draft -> s_edit [label="Passes Draft"];
            s_edit -> s_audit [label="Passes Edited Doc"];
        }

        subgraph cluster_parallel_pattern {
            label="Pattern C: Parallel Scatter-Gather";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            p_split [label="Scatter Node", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            p_worker1 [label="Worker 1 (Market A)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            p_worker2 [label="Worker 2 (Market B)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            p_join [label="Gather / Synthesize", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
            p_split -> p_worker1 [label="Async"];
            p_split -> p_worker2 [label="Async"];
            p_worker1 -> p_join [label="Joins"];
            p_worker2 -> p_join [label="Joins"];
        }
    }
    """
    st.graphviz_chart(patterns_dot, use_container_width=True)

# --- SECTION 2: SUPERVISOR AGENT ---
st.header("2. Supervisor Agent", divider="rainbow")
st.markdown("""
The **Supervisor Agent** pattern is the most widely deployed enterprise pattern. A central coordinating agent acts as the
orchestrator, analyzing user goals, decomposing tasks, assigning subtasks to worker agents, inspecting results,
and determining when the mission is fulfilled.
""")

col_sup1, col_sup2 = st.columns(2)
with col_sup1:
    st.markdown("""
    #### Architectural Mechanics
    - **Dynamic Routing**: The supervisor uses structured output (e.g. `with_structured_output(RouterDecision)`) to select the next agent from an enumerated list (`Literal["researcher", "coder", "FINISH"]`).
    - **Quality Control**: If a worker agent returns an error or incomplete artifact, the supervisor detects the shortfall and re-dispatches with refined instructions.
    - **Single Point of Egress**: The supervisor synthesizes worker outputs into the final response delivered to the end-user.
    """)

with col_sup2:
    with st.container(border=True):
        st.markdown("📋 **Supervisor Schema in Pydantic**")
        st.code('''from pydantic import BaseModel, Field
from typing import Literal

class SupervisorDecision(BaseModel):
    # Routing schema emitted by Supervisor LLM
    next_agent: Literal["research_expert", "code_generator", "security_auditor", "FINISH"] = Field(
        description="The next specialist agent to call, or FINISH if the user request is satisfied."
    )
    instructions: str = Field(
        description="Granular instructions directing the chosen specialist's action."
    )''', language="python")

# --- SECTION 3: ROUTER AGENT ---
st.header("3. Router Agent", divider="rainbow")
st.markdown("""
The **Router Agent** is a lightweight, one-way decision maker. Unlike the Supervisor (which maintains an iterative feedback loop),
a Router evaluates ingress user intent once and dispatches control to exactly one downstream specialist or chain.
""")

col_rt1, col_rt2 = st.columns(2)
with col_rt1:
    st.markdown("""
    #### Characteristics of Routers
    - **Zero Execution Loop**: One-shot execution. The router does not wait for workers to return.
    - **High Throughput / Low Latency**: Operates with small fast models (e.g., GPT-4o-mini, Claude 3.5 Haiku) classifying intent into pre-indexed routes.
    - **Implementation in LCEL**: Constructed cleanly using `RunnableBranch` or conditional function routers.
    """)

with col_rt2:
    with st.container(border=True):
        st.markdown("⚡ **LCEL Runnable Routing Pattern**")
        st.code("""
def route_intent(info: dict):
    if "code" in info["topic"]:
        return code_agent_chain
    elif "sql" in info["topic"]:
        return sql_agent_chain
    else:
        return general_faq_chain

full_chain = (
    {"topic": classifier_chain, "query": RunnablePassthrough()}
    | RunnableLambda(route_intent)
)
        """, language="python")

# --- SECTION 4: WORKER AGENTS ---
st.header("4. Worker Agents", divider="rainbow")
st.markdown("""
**Worker Agents** are task-execution units. A worker possesses no authority over global mission lifecycle;
it accepts a task description, executes necessary tool calls, formats its results, and returns control back to the caller.
""")

st.markdown("""
#### The Three Golden Rules for Worker Agents:
1. **Bounded State**: Workers must not mutate global settings or overwrite other workers' metadata.
2. **Explicit Returns**: A worker must always emit a definitive artifact or error message so the supervisor is never left waiting.
3. **No Tool Hoarding**: A worker should possess only the tools strictly required for its mandate (e.g., a SQL worker gets database tools, but zero shell or web tools).
""")

# --- SECTION 5: SPECIALIST AGENTS ---
st.header("5. Specialist Agents", divider="rainbow")
st.markdown("""
A **Specialist Agent** is an extension of the worker concept equipped with a deep persona, domain-specific terminology,
and specialized knowledge.
""")

s_c1, s_c2, s_c3 = st.columns(3)
with s_c1:
    with st.container(border=True):
        st.markdown("🔬 **The Research Specialist**")
        st.caption("Persona: Academic Investigator")
        st.markdown("Trained to parse arxiv papers, search Google Scholar, cross-check claims, and output IEEE-style citations.")

with s_c2:
    with st.container(border=True):
        st.markdown("💻 **The Python Coder Specialist**")
        st.caption("Persona: Senior Systems Architect")
        st.markdown("Trained to write type-hinted PEP 8 code, execute unit tests, and output git patches.")

with s_c3:
    with st.container(border=True):
        st.markdown("🛡️ **The Security Specialist**")
        st.caption("Persona: Infosec Pen-Tester")
        st.markdown("Trained to inspect code for OWASP Top 10 vulnerabilities, PII leakage, and SQL injection flaws.")

# --- SECTION 6: SEQUENTIAL AGENTS ---
st.header("6. Sequential Agents", divider="rainbow")
st.markdown("""
The **Sequential Pipeline** pattern (Assembly Line) chains agents linearly. The output of Agent A becomes the direct
input to Agent B, which subsequently feeds Agent C.
""")

seq_dot = """
digraph SequentialPipeline {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    input [label="User Outline", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    draft_agent [label="1. Content Drafter\\nGenerates 1,000-word draft", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    fact_agent [label="2. Fact Checker\\nVerifies statistics & dates", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    polish_agent [label="3. Copy Editor\\nRefines tone & formatting", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    final_post [label="Published Article", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];

    input -> draft_agent [label="Prompt"];
    draft_agent -> fact_agent [label="Draft Text"];
    fact_agent -> polish_agent [label="Verified Text"];
    polish_agent -> final_post [label="Final Output"];
}
"""
st.graphviz_chart(seq_dot, use_container_width=True)

st.markdown("""
**When to Use**: Predictable, staged transformation processes where step order is strictly deterministic
(e.g., Translation -> Proofreading -> Legal Review).
""")

# --- SECTION 7: PARALLEL AGENTS ---
st.header("7. Parallel Agents", divider="rainbow")
st.markdown("""
The **Parallel Scatter-Gather** pattern executes multiple worker agents concurrently. When a complex research task
can be divided into independent sub-problems (e.g., analyzing 4 competitors simultaneously), running agents in parallel
dramatically reduces wall-clock latency.
""")

par_dot = """
digraph ParallelScatterGather {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    target [label="Research Topic: 'AI Chips'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    worker_nvidia [label="Agent 1: Nvidia Architecture", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    worker_amd [label="Agent 2: AMD Instinct Series", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    worker_google [label="Agent 3: Google TPU v5e", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    synthesizer [label="Synthesizer Agent\\nGenerates Comparative Report", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];

    target -> worker_nvidia [label="Async Branch 1"];
    target -> worker_amd [label="Async Branch 2"];
    target -> worker_google [label="Async Branch 3"];
    worker_nvidia -> synthesizer [label="Report 1"];
    worker_amd -> synthesizer [label="Report 2"];
    worker_google -> synthesizer [label="Report 3"];
}
"""
st.graphviz_chart(par_dot, use_container_width=True)

st.markdown("""
**Implementation in LCEL**: Built using `RunnableParallel`.
All branches execute asynchronously across worker threads or coroutines, joining their outputs into an aggregation dictionary.
""")

# --- SECTION 8: HIERARCHICAL AGENTS ---
st.header("8. Hierarchical Agents", divider="rainbow")
st.markdown("""
When a system scales to 10+ agents, a single supervisor becomes overloaded. The **Hierarchical Multi-Agent Pattern**
introduces multi-tier management: an Executive Director delegates to Department Supervisors, who in turn supervise
granular worker agents.
""")

hier_dot = """
digraph HierarchicalArchitecture {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    exec_agent [label="Executive Director Agent\\nManages high-level project milestones", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    lead_dev [label="Software Lead Supervisor", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    lead_research [label="Research Lead Supervisor", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    dev_frontend [label="Frontend Coder", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    dev_backend [label="Backend Coder", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    res_web [label="Web Scraper", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    res_db [label="Database Analyst", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];

    exec_agent -> lead_dev [label="Assigns Build"];
    exec_agent -> lead_research [label="Assigns Study"];
    lead_dev -> dev_frontend [label="Manages"];
    lead_dev -> dev_backend [label="Manages"];
    lead_research -> res_web [label="Manages"];
    lead_research -> res_db [label="Manages"];
}
"""
st.graphviz_chart(hier_dot, use_container_width=True)

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Supervisor-Worker Architecture in LCEL", divider="rainbow")
st.markdown("""
Below is a runnable implementation of a Supervisor-Worker architecture using LangChain LCEL runnables
and typed Pydantic routing.
""")

prod_pattern_code = '''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# 1. Structured Routing Output Schema
class RoutePlan(BaseModel):
    next_step: Literal["researcher", "coder", "FINISH"] = Field(
        description="Next worker agent to invoke, or FINISH if the goal is met."
    )
    task: str = Field(description="Instructions for the target agent.")

# 2. Worker Functions
def research_worker(task_spec: str) -> str:
    # In production: invokes model.bind_tools([search_tool])
    return f"Research Artifact: Gathered intel on '{task_spec}' successfully."

def coder_worker(task_spec: str) -> str:
    # In production: invokes model.bind_tools([code_repl])
    return f"Code Artifact: Implemented solution for '{task_spec}'."

# 3. Supervisor Dispatch Controller
class SupervisorPipeline:
    def __init__(self):
        self.state = {"history": [], "artifacts": []}

    def run(self, user_mission: str) -> dict:
        self.state["history"].append(f"Mission: {user_mission}")
        
        # Iteration loop with hard safety cap
        for iteration in range(5):
            # Simulated Supervisor Decision (In prod: model.with_structured_output(RoutePlan))
            if iteration == 0:
                decision = RoutePlan(next_step="researcher", task="Investigate Vector DBs")
            elif iteration == 1:
                decision = RoutePlan(next_step="coder", task="Build ChromaDB retrieval script")
            else:
                decision = RoutePlan(next_step="FINISH", task="All milestones complete.")

            if decision.next_step == "FINISH":
                break
            elif decision.next_step == "researcher":
                result = research_worker(decision.task)
                self.state["artifacts"].append(result)
            elif decision.next_step == "coder":
                result = coder_worker(decision.task)
                self.state["artifacts"].append(result)

        return self.state
'''
st.code(prod_pattern_code, language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Multi-Agent Pattern Simulator", divider="rainbow")
st.markdown("""
Simulate how different architectural patterns handle workloads of varying scale.
Evaluate how Supervisor, Sequential, and Parallel topologies impact total LLM invocations and latency.
""")

pattern_engine = MultiAgentPatternEngine()

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.subheader("⚙️ Workload & Pattern Settings")
    chosen_pattern = st.selectbox(
        "Choose Multi-Agent Architecture Pattern:",
        ["Supervisor", "Sequential", "Parallel (Scatter-Gather)", "Hierarchical"]
    )
    workload_tasks = st.slider("Number of Subtasks to Complete:", min_value=2, max_value=8, value=4)
    run_pattern_sim = st.button("🚀 Evaluate Pattern Performance", type="primary", use_container_width=True)

with col_p2:
    st.subheader("📊 Architectural Performance Metrics")
    if run_pattern_sim:
        metrics = pattern_engine.simulate_pattern(chosen_pattern, workload_tasks)
        
        st.markdown(
            f"<div class='pattern-card'><span class='badge-pattern-active'>{chosen_pattern.upper()}</span>"
            f"<br><br><b>Topology Mechanism:</b> {metrics['description']}</div>",
            unsafe_allow_html=True
        )

        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("Total LLM Calls Required", metrics["total_calls"])
            st.metric("Supports Concurrency", "YES (Concurrent)" if metrics["supports_concurrency"] else "NO (Sequential)")
        with col_met2:
            st.metric("Estimated Latency Index", f"{metrics['latency_index_seconds']}s")
            
        st.info("💡 **Architectural Trade-off**: Parallel execution slashes wall-clock latency for independent tasks. Supervisor execution provides the highest accuracy and self-correction but incurs higher LLM invocation overhead.")
    else:
        st.info("Select a pattern and subtask volume, then click 'Evaluate Pattern Performance'.")

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: When should you select a Parallel (Scatter-Gather) pattern over a Supervisor pattern?"):
    st.markdown("""
    **Answer**: Select Parallel (Scatter-Gather) when the subtasks are completely independent of one another and can be computed
    simultaneously (e.g. running 5 different financial valuation models on the same company dataset). Select Supervisor when subtasks
    have complex sequential dependencies where the outcome of step 2 depends on the findings of step 1, or where ongoing quality control
    and adaptive replanning are required.
    """)

with st.expander("Q2: What is the main architectural danger of Sequential (Assembly-Line) agent pipelines?"):
    st.markdown("""
    **Answer**: Cascading failure and error compounding. If Agent 1 in a 4-agent sequential pipeline makes an erroneous assumption
    or hallucination, Agent 2 treats that error as established ground truth, compounding the mistake. By the time the document reaches
    Agent 4, the output is completely corrupted. Mitigate this by embedding validation checkpoints between sequential nodes.
    """)

with st.expander("Q3: How does a Hierarchical Agent architecture prevent Supervisor bottlenecking?"):
    st.markdown("""
    **Answer**: In flat supervisor systems with 10+ workers, the supervisor prompt becomes bloated with too many tool/agent definitions,
    leading to routing mistakes. A Hierarchical tree divides authority: an Executive Director only routes between Department Supervisors
    (e.g., Code Lead, Research Lead), and each Department Supervisor manages 2 to 3 domain workers. This bounds the cognitive load at every tier.
    """)

with st.expander("Q4: Why should Router Agents use lightweight classification models rather than frontier LLMs?"):
    st.markdown("""
    **Answer**: Router agents perform narrow intent classification (e.g., mapping a query to 1 of 5 categories). Using expensive frontier
    models (like GPT-4o) adds unnecessary financial cost and 1-2 seconds of user-perceived latency. Fast, small models (such as GPT-4o-mini
    or fine-tuned local models) achieve 99%+ routing accuracy with sub-200ms response times.
    """)

with st.expander("Q5: What termination safeguards are required in Supervisor agent loops?"):
    st.markdown("""
    **Answer**: Without deterministic guards, a supervisor can loop indefinitely between confused workers. Production systems mandate:
    (1) A hard recursion limit (`recursion_limit=15`); (2) An explicit `FINISH` action in the supervisor routing schema;
    (3) Loop-detection heuristics that trigger an automatic fallback if the supervisor selects the exact same worker 3 times consecutively.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Uncapped Supervisor Loops)**: Omitting a hard iteration ceiling on the supervisor routing loop. If a worker fails to satisfy the supervisor, the system burns infinite tokens and runs forever.")
    st.warning("**Trap 2 (State Clobbering in Parallel Joins)**: Multiple parallel agents writing to the exact same key in a shared dictionary without a conflict-resolution reducer, leading to race-condition data loss.")
with col_trap2:
    st.error("**Trap 3 (Single-Point-of-Failure Supervisor)**: Relying on a fragile, poorly prompted supervisor that hallucinates agent names not present in the system, causing runtime KeyError crashes.")
    st.warning("**Trap 4 (Sequential Latency Accumulation)**: Chaining 6 agents sequentially when 4 of them could run concurrently in parallel, turning a 3-second user interaction into a 25-second wait.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Supervisor Pattern in MAS?")
        st.caption("Flip for answer:")
        st.info("**Back**: A centralized orchestrator LLM that iteratively assigns subtasks to specialist workers, inspects artifacts, and terminates the mission.")
    with st.container(border=True):
        st.markdown("**Front**: How does Scatter-Gather parallelization work?")
        st.caption("Flip for answer:")
        st.info("**Back**: It fans out independent tasks to run concurrently across multiple workers, then aggregates all results into a single synthesizer node.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary risk of Sequential agent pipelines?")
        st.caption("Flip for answer:")
        st.info("**Back**: Cascading error compounding: early hallucinations propagate downstream and corrupt all subsequent agent stages.")
    with st.container(border=True):
        st.markdown("**Front**: When is a Hierarchical multi-agent architecture justified?")
        st.caption("Flip for answer:")
        st.info("**Back**: When a team grows beyond 6-8 agents, requiring tiered leadership (Director -> Team Leads -> Workers) to prevent supervisor cognitive overload.")

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural differences between Supervisor, Router, and Hierarchical patterns.", value=True)
c2 = st.checkbox("I know how to implement structured routing schemas using Pydantic in LCEL.", value=True)
c3 = st.checkbox("I understand when to utilize Parallel Scatter-Gather over Sequential pipelines.", value=True)
c4 = st.checkbox("I recognize how to prevent cascading errors in sequential agent chains.", value=True)
c5 = st.checkbox("I know how to enforce hard iteration limits to avoid infinite supervisor loops.", value=True)
c6 = st.checkbox("I understand how multi-tier hierarchical delegation bounds cognitive load.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 26.3: Advanced Multi-Agent Systems.")
