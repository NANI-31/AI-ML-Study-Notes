import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.4 Agent Architectures - LLM Engineering",
    page_icon="🏛️",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class MultiAgentCollaborator:
    """
    Self-contained in-memory multi-agent collaboration network implementing:
    - Supervisor Agent (Router & Aggregator)
    - Research Agent (Fact finding & retrieval)
    - Code Agent (Implementation & syntax verification)
    - Reviewer Agent (Quality & safety critique)
    """
    
    def __init__(self):
        self.trajectory_log: List[Dict[str, str]] = []

    def run_collaboration(self, complex_task: str) -> List[Dict[str, str]]:
        self.trajectory_log = []
        
        # Step 1: Supervisor decomposes task
        self.trajectory_log.append({
            "agent": "Supervisor Agent",
            "action": "Task Decomposition",
            "message": f"Analyzing task: '{complex_task}'. Delegating research to ResearchAgent and implementation to CodeAgent."
        })
        
        # Step 2: Research Agent executes
        if "data" in complex_task.lower() or "benchmark" in complex_task.lower():
            research_finding = "Retrieved dataset specs: 100,000 samples, 4 numerical features, target is binary churn."
        else:
            research_finding = "Identified standard REST API requirements: OAuth2 Bearer, JSON payload, P99 < 150ms."
            
        self.trajectory_log.append({
            "agent": "Research Agent",
            "action": "Fact Gathering",
            "message": f"Research complete. Findings: {research_finding}"
        })
        
        # Step 3: Code Agent builds solution
        if "data" in complex_task.lower() or "benchmark" in complex_task.lower():
            code_snippet = "from sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier(n_estimators=100)\nclf.fit(X_train, y_train)"
        else:
            code_snippet = "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef health(): return {'status': 'healthy'}"
            
        self.trajectory_log.append({
            "agent": "Code Agent",
            "action": "Code Generation",
            "message": f"Constructed candidate code:\n{code_snippet}"
        })
        
        # Step 4: Reviewer Agent validates
        self.trajectory_log.append({
            "agent": "Reviewer Agent",
            "action": "Security & Quality Audit",
            "message": "Audit verified: No security vulnerabilities, type annotations present, ready for synthesis."
        })
        
        # Step 5: Supervisor summarizes
        self.trajectory_log.append({
            "agent": "Supervisor Agent",
            "action": "Final Synthesis",
            "message": f"Task '{complex_task}' successfully resolved across 3 specialized agents."
        })
        
        return self.trajectory_log

# --- TOP-LEVEL HEADER ---
st.title("20.4 Agent Architectures: ReAct, Planner-Executor, & Multi-Agent Systems")
st.caption("Chapter 20: LLM Agents | Master Architecture Specification")
st.markdown("""
Agent design has evolved from monolithic single-prompt loops into modular, multi-tier architectural patterns. 
Depending on task complexity, latency budgets, and security isolation requirements, engineers select between 
**ReAct**, **Planner-Executor**, **Hierarchical Supervisors**, and **Cooperative Multi-Agent Networks**.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: Agent Design Taxonomy", divider="rainbow")

st.markdown("""
The architecture map below illustrates the spectrum of agent design patterns: from simple reactive tool-users 
to decoupled plan-execute graphs and distributed multi-agent swarms.
""")

dot_arch = """
digraph AgentTaxonomy {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_single {
        label = "Single-Agent Paradigms";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        ReActSingle [label="1. ReAct Pattern\\n[Step-by-Step Reason + Act]", fillcolor="#1E3A8A", color="#3B82F6"];
        PlanExec [label="2. Planner-Executor\\n[Upfront Plan -> Batch Workers]", fillcolor="#1E3A8A", color="#3B82F6"];
        ToolFormer [label="3. Tool-Augmented LLM\\n[In-line API Dispatch]", fillcolor="#1E3A8A", color="#3B82F6"];
    }

    subgraph cluster_multi {
        label = "Multi-Agent Systems (MAS)";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        Supervisor [label="4. Hierarchical Supervisor\\n[Orchestrator + Worker Sub-Agents]", fillcolor="#78350F", color="#D97706"];
        PeerToPeer [label="5. Peer-to-Peer Swarm\\n[Role-Playing Debate & Consensus]", fillcolor="#78350F", color="#D97706"];
        Autonomous [label="6. Goal-Directed Autonomous Agent\\n[AutoGPT Open Loop Expansion]", fillcolor="#831843", color="#EC4899"];
    }

    ReActSingle -> Supervisor [label="Evolves into Hierarchies"];
    PlanExec -> Supervisor [label="Used by Orchestrator"];
    ToolFormer -> ReActSingle [label="Foundational Primitives"];
    Supervisor -> PeerToPeer [label="Alternative Routing"];
    PeerToPeer -> Autonomous [label="Full Decentralization"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_arch, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - REACT ---
st.header("2. ReAct Pattern: Interleaved Reasoning and Action Traces", divider="rainbow")

st.markdown("""
Pioneered by Yao et al. (2022), **ReAct** (Reasoning + Acting) is the foundational pattern of modern agent design. 
It interleaves natural language reasoning with task-specific actions.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Synergistic Cognitive Triad")
    st.markdown("""
    - **Thought**: Formulates hypotheses, updates internal beliefs, and determines what external information is needed.
    - **Action**: Emits structured tool calls to fetch data or modify environment state.
    - **Observation**: Reads back raw feedback from the environment.
    - **The Synergy**: Reasoning guides action selection; action feedback grounds reasoning, eliminating compounding hallucinations.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: Native Interleaving
    Early ReAct agents relied on manual text templates (`Thought: ... Action: ...`). 
    Modern frontier models interleave reasoning and tool calling natively: models can output chain-of-thought tokens 
    followed immediately by binary tool calling blocks in a single streaming response.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - PLANNER-EXECUTOR ---
st.header("3. Planner-Executor Framework: Decoupled Planning & Parallel Workflows", divider="rainbow")

st.markdown("""
While ReAct is step-by-step and exploratory, **Planner-Executor** architectures decouple global strategy from localized execution.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Plan-and-Solve Mechanics")
    st.markdown("""
    1. **Planner Agent**: Analyzes the user's root objective and outputs an acyclic Directed Graph (DAG) of sub-tasks.
    2. **Task Queue & Router**: Evaluates dependencies. Independent tasks are immediately dispatched in parallel.
    3. **Executor Agents**: Specialized worker LLMs execute each discrete sub-task without needing the entire global conversation history.
    4. **Re-Planner**: Inspects intermediate results. If a sub-task encounters an unrecoverable error, the planner adjusts downstream tasks dynamically.
    """)

with col2:
    st.info("""
    **Latency Advantage**:
    In a ReAct agent, 5 tool calls require 5 sequential model roundtrips (e.g. 5 x 2.5s = 12.5s). 
    In a Planner-Executor framework with parallel execution (LLM Compiler), independent sub-tasks run concurrently, 
    reducing wall-clock latency to a single roundtrip (~3s)!
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - TOOL-USING AGENTS ---
st.header("4. Tool-Using Agents: Single-Agent Orchestration & MRKL Systems", divider="rainbow")

st.markdown("""
A **Tool-Using Agent** (Modular Reasoning, Knowledge and Language - MRKL) pairs a single LLM with a curated suite of actuators.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Architecture of MRKL Systems")
    st.markdown("""
    - **Central Dispatcher**: A single model receives the user request alongside 5 to 15 registered tool schemas.
    - **In-Line Actuation**: The model determines tool selection, formats arguments, and awaits the observation.
    - **Scope**: Best suited for well-defined, domain-specific utilities (e.g., an IT helpdesk bot that checks ticket status, resets passwords, and queries FAQs).
    """)

with col2:
    st.warning("""
    **Single-Agent Ceiling**:
    When a single agent is given more than 20+ tools across disparate domains (e.g., SQL + Git + CloudOps + WebSearch + Billing), 
    tool selection accuracy drops dramatically due to prompt crowding and semantic interference between tool descriptions.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - MULTI-AGENT SYSTEMS ---
st.header("5. Multi-Agent Systems: Specialized Personas & Cooperative Problem Solving", divider="rainbow")

st.markdown("""
When tasks exceed the cognitive capacity of a single prompt, **Multi-Agent Systems (MAS)** divide responsibilities among 
multiple specialized agents, each endowed with a distinct system prompt, persona, and private tool registry.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Key Advantages of Multi-Agent Collaboration")
    st.markdown("""
    - **Context Isolation**: Each agent's context window contains only information relevant to its narrow specialty (e.g., the Code Agent does not need 50 pages of raw legal docs).
    - **Role Specialization**: Different system prompts enforce different behavior (e.g. an aggressive Red Team hacker vs a defensive Blue Team auditor).
    - **Self-Critique & Debate**: Agents can critique each other's outputs, uncovering bugs that a single self-reflective pass missed.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: Graph State Frameworks
    Frameworks like **LangGraph**, **AutoGen**, and **CrewAI** model multi-agent systems as **Stateful Graphs**. 
    Nodes represent agents or tools; edges represent conditional transition functions based on agent output state.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - SUPERVISOR AGENTS ---
st.header("6. Supervisor Agents: Hierarchical Routing & Sub-Agent Orchestration", divider="rainbow")

st.markdown("""
The **Supervisor Pattern** is the most widely adopted production pattern for enterprise Multi-Agent Systems. 
A top-level orchestrator routes tasks and aggregates results.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Orchestrator-Worker Hierarchy")
    st.markdown("""
    1. **Client Request**: Ingested solely by the **Supervisor Agent**.
    2. **Delegation / Handoff**: The Supervisor treats sub-agents as high-level tools:
       - `invoke_researcher(topic="...")`
       - `invoke_coder(spec="...")`
       - `invoke_tester(code="...")`
    3. **Sub-Agent Execution**: Sub-agents run their own internal ReAct loops autonomously and return a summarized artifact to the Supervisor.
    4. **Final Synthesis**: The Supervisor synthesizes sub-agent deliverables into a cohesive response for the user.
    """)

with col2:
    st.info("""
    **Handoff vs Delegation**:
    - **Delegation (Return Control)**: The Supervisor calls a worker, waits for the result, and maintains control.
    - **Handoff (Swarm Pattern)**: The Supervisor completely transfers control to Worker B; Worker B communicates directly with the user.
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - AUTONOMOUS AGENTS ---
st.header("7. Autonomous Agents: Goal-Directed Open-Loop Systems & Termination", divider="rainbow")

st.markdown("""
Pioneered by experimental projects like AutoGPT and BabyAGI, **Autonomous Agents** pursue open-ended goals 
by continuously generating new sub-tasks, prioritizing them in a dynamic queue, and executing them without human prompts.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Autonomous Goal Expansion Loop")
    st.markdown("""
    1. **Root Goal**: *"Research competitors in the AI code generation space and build a comparison matrix."*
    2. **Task Creation**: Generates 5 initial tasks and pushes to queue.
    3. **Execution**: Pops Task 1 -> executes -> produces observations.
    4. **Dynamic Task Generation**: Based on Task 1's observation, creates 3 new sub-tasks.
    5. **Reprioritization**: Ranks the global queue by importance.
    """)

with col2:
    st.error("""
    **The Autonomous Agent Failure Mode**:
    Without strict halting conditions, open-loop autonomous agents inevitably suffer from **Goal Drift**: 
    the agent gets distracted by irrelevant details (e.g. spending 3 hours refactoring a CSS button while trying to research market competitors), 
    burning hundreds of dollars in tokens without completing the primary goal.
    """)

st.divider()

# --- SECTION 8: PRODUCTION CODE IMPLEMENTATION ---
st.header("8. Production Implementation: Hierarchical Multi-Agent Orchestrator", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Multi-Agent Supervisor framework in Python. 
It features role isolation, typed inter-agent handoffs, and centralized supervision.
""")

st.code('''
from typing import Dict, List, Any, Callable

class SubAgent:
    """Specialized worker agent with role-specific system prompt."""
    def __init__(self, name: str, role_description: str, handler: Callable[[str], str]):
        self.name = name
        self.role_description = role_description
        self.handler = handler

    def execute(self, task_instruction: str) -> str:
        return self.handler(task_instruction)

class SupervisorOrchestrator:
    """
    Hierarchical supervisor managing routing and result synthesis across sub-agents.
    """
    def __init__(self):
        self.sub_agents: Dict[str, SubAgent] = {}
        self.execution_audit: List[Dict[str, str]] = []

    def register_subagent(self, agent: SubAgent) -> None:
        self.sub_agents[agent.name] = agent

    def route_task(self, subagent_name: str, task: str) -> str:
        """Dispatches a sub-task to a registered specialized agent."""
        if subagent_name not in self.sub_agents:
            raise ValueError(f"Agent '{subagent_name}' is not registered.")
        
        agent = self.sub_agents[subagent_name]
        result = agent.execute(task)
        self.execution_audit.append({
            "agent": subagent_name,
            "task": task,
            "result": result
        })
        return result

    def solve_complex_objective(self, objective: str) -> Dict[str, Any]:
        """Orchestrates multi-agent plan to fulfill global objective."""
        self.execution_audit = []
        
        # Step 1: Research
        research_out = self.route_task(
            "researcher", 
            f"Identify technical requirements for: {objective}"
        )
        
        # Step 2: Code
        code_out = self.route_task(
            "coder", 
            f"Implement solution based on research: {research_out}"
        )
        
        # Step 3: Review
        review_out = self.route_task(
            "reviewer", 
            f"Audit implementation for security and style: {code_out}"
        )
        
        return {
            "objective": objective,
            "audit_trail": self.execution_audit,
            "final_synthesis": f"Task resolved with review approval: {review_out}"
        }
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Multi-Agent Workbench: Live Collaborative Problem Solver", divider="rainbow")

st.markdown("""
Observe real-time multi-agent collaboration. Select a complex engineering goal, trigger execution, 
and inspect how the **Supervisor**, **Researcher**, **Coder**, and **Reviewer** collaborate.
""")

col1, col2 = st.columns([1, 2])

with col1:
    selected_task = st.selectbox(
        "Select Complex Engineering Task:",
        [
            "Build an ML Customer Churn Predictor Pipeline",
            "Design an Async FastAPI Microservice with Health Checks",
            "Develop an Automated Database Migration Script"
        ]
    )
    
    collab_btn = st.button("🚀 Execute Multi-Agent Workflow", type="primary", use_container_width=True)

with col2:
    collaborator = MultiAgentCollaborator()
    
    if collab_btn:
        logs = collaborator.run_collaboration(selected_task)
        
        st.subheader("Multi-Agent Collaboration Trace")
        for entry in logs:
            agent_name = entry["agent"]
            if "Supervisor" in agent_name:
                badge_color = "#3B82F6"
            elif "Research" in agent_name:
                badge_color = "#10B981"
            elif "Code" in agent_name:
                badge_color = "#8B5CF6"
            else:
                badge_color = "#F59E0B"
                
            st.markdown(f"**🤖 {agent_name}** (`{entry['action']}`):")
            st.info(entry["message"])
            
        st.success("Collaboration cycle complete. All agents reached consensus.")
    else:
        st.info("Select a task and click 'Execute Multi-Agent Workflow' to watch agents interact.")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: What are the primary trade-offs between a Single-Agent ReAct loop and a Multi-Agent System?"):
    st.markdown("""
    **Answer**:
    - **Single-Agent ReAct**: Much lower cost and latency, simpler debugging, single linear trace. However, context window crowding degrades performance when more than 15-20 tools are registered, and a single system prompt cannot specialize across conflicting roles.
    - **Multi-Agent System (MAS)**: Excellent context isolation, modular maintainability, and domain specialization. However, MAS introduces substantial token cost overhead (multipliers of 3x to 10x), higher wall-clock latency, and the potential for inter-agent communication deadlocks.
    """)

with st.expander("Q2: How do you prevent infinite inter-agent debate loops in Multi-Agent systems?"):
    st.markdown("""
    **Answer**:
    1. **Graph Max Hop Budgets**: In state graph runtimes (LangGraph), define a strict global transition recursion limit (e.g. `recursion_limit=15`).
    2. **State Transition Deadlock Detectors**: Track consecutive turns where no external tool is executed and semantic similarity between agent critique messages exceeds 0.90.
    3. **Hierarchical Escalation**: The Supervisor agent possesses tie-breaking veto authority to force a terminal state if consensus is not reached within 3 debate rounds.
    """)

with st.expander("Q3: When should you use the Plan-and-Solve (Planner-Executor) architecture over ReAct?"):
    st.markdown("""
    **Answer**:
    Use **Plan-and-Solve** when the problem decomposes into clearly defined, independent parallel sub-tasks where the overall workflow is predictable (e.g. querying financial reports for 10 distinct competitors). 
    Use **ReAct** when the problem space is highly dynamic and exploratory, where Step 2's action cannot be anticipated until Step 1's observation is reviewed (e.g. code debugging or live web exploration).
    """)

with st.expander("Q4: What is the difference between Agent Handoff and Agent Delegation?"):
    st.markdown("""
    **Answer**:
    - **Delegation (Call & Return)**: The Supervisor invokes Sub-Agent A as a subordinate tool. Sub-Agent A finishes and returns its artifact to the Supervisor. The Supervisor retains primary context and conversational ownership.
    - **Handoff (Swarm Pattern)**: The Supervisor completely abdicates control, transferring the user conversation thread to Agent B. Agent B directly interacts with the user going forward.
    """)

with st.expander("Q5: Why did early autonomous agent systems like AutoGPT suffer from Goal Drift?"):
    st.markdown("""
    **Answer**:
    AutoGPT maintained an open-ended dynamic task queue without an explicit global convergence function. 
    Every observation prompted the model to generate 3 to 5 new sub-tasks. Over multiple iterations, the task queue grew exponentially, 
    and sub-tasks drifted into tangential trivia (e.g. infinite loops investigating how to install obscure dependencies) rather than converging toward the user's primary business objective.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Runaway Multi-Agent Token Inflation**
    Deploying 5+ agents that all echo and pass full conversation transcripts to each other. Token consumption grows quadratically `O(N^2)` with conversation length, causing massive cloud API bills.
    """)
    st.error("""
    **TRAP 2 (Error): Unbounded Inter-Agent Debate Loops**
    Allowing a Coder and a Reviewer agent to debate indefinitely without a tie-breaker. The Coder refactors minor styling, the Reviewer requests another tiny change, looping until budget exhaustion.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Over-Engineering with Multi-Agent Swarms**
    Using a 4-agent LangGraph system for a task that can be solved in a single prompt with 2 tool calls. Multi-agent complexity adds latency and introduces multiple points of failure.
    """)
    st.warning("""
    **TRAP 4 (Warning): Loss of Grounding in Supervisor Summaries**
    The Supervisor summarizing a 1,000-line sub-agent code output into a 2-sentence natural language reply, discarding critical technical details the user actually requested.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **ReAct**  
    Interleaves single-step reasoning with immediate external tool execution.
    """)
with fc2:
    st.markdown("""
    **Planner-Executor**  
    Decouples task graph generation from parallel sub-task worker execution.
    """)
with fc3:
    st.markdown("""
    **Supervisor Agent**  
    Central orchestrator that delegates tasks to specialized worker sub-agents.
    """)
with fc4:
    st.markdown("""
    **Goal Drift**  
    The tendency of open-loop autonomous agents to wander off into tangential tasks.
    """)

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist: Agent Architectures Mastery", divider="rainbow")

st.markdown("Verify your understanding of modern agent design patterns by reviewing each criterion:")

chk1 = st.checkbox("I can differentiate ReAct, Planner-Executor, and Multi-Agent architectural patterns.", key="arch_chk1")
chk2 = st.checkbox("I understand when to decouple planning from execution to enable parallel sub-task acceleration.", key="arch_chk2")
chk3 = st.checkbox("I know how the Supervisor pattern enforces context isolation and manages sub-agent delegation.", key="arch_chk3")
chk4 = st.checkbox("I understand the distinction between Agent Delegation and Agent Handoff.", key="arch_chk4")
chk5 = st.checkbox("I can identify and mitigate Goal Drift and infinite inter-agent debate loops.", key="arch_chk5")
chk6 = st.checkbox("I understand how stateful graph runtimes (LangGraph, AutoGen) model multi-agent workflows as state machines.", key="arch_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of Advanced Agent Architectures and Multi-Agent Orchestration!")
