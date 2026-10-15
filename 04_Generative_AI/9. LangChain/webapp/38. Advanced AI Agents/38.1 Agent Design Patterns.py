import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="38.1 Agent Design Patterns - LangChain Systems",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for design pattern telemetry
st.markdown("""
<style>
    .pattern-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-react {
        background-color: #0284C7;
        color: #F0F9FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-plan {
        background-color: #7C3AED;
        color: #F5F3FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-critic {
        background-color: #D97706;
        color: #FFFBEB;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .step-box {
        border-left: 3px solid #38BDF8;
        padding-left: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class AgentDesignPatternSimulator:
    """
    Simulates execution traces across primary cognitive agent patterns:
    ReAct, Router, Planner-Executor, Supervisor, Reflection, Critic, and Self-Correction.
    """
    def __init__(self):
        self.preset_traces = {
            "ReAct: Diagnostic System Troubleshooting": {
                "pattern": "ReAct (Reasoning + Action)",
                "goal": "Diagnose intermittent 504 Gateway Timeouts on production payment endpoint.",
                "steps": [
                    {"thought": "Need to inspect ingress latency logs over the past 60 minutes.", "action": "query_datadog_logs(service='payment-api', window='60m')", "observation": "Observed 42 timeout events correlating with downstream database connection pool exhaustion."},
                    {"thought": "Database pool exhaustion detected. Need to check active database connections and replica lag.", "action": "inspect_db_metrics(cluster='aurora-prod-pg')", "observation": "Primary node connection count reached 100% capacity (500/500). Read replicas idle at 12% load."},
                    {"thought": "Root cause identified: Read queries are hitting the primary node rather than read-replicas due to stale connection string configuration.", "action": "conclude_root_cause()", "observation": "Remediation recommended: Point read-heavy billing queries to Aurora Read-Replica endpoint."}
                ],
                "outcome": "Identified database pool saturation on write-primary caused by unrouted read traffic.",
                "efficiency_score": 92
            },
            "Planner-Executor: Infrastructure Migration": {
                "pattern": "Planner-Executor (Plan-and-Solve)",
                "goal": "Migrate 25 TB Vector Database from single-node instance to distributed multi-tenant cluster.",
                "steps": [
                    {"thought": "Generating step-by-step migration plan.", "action": "planner.generate_plan()", "observation": "Plan: 1. Provision cluster -> 2. Mirror CDC write stream -> 3. Bulk dump historical vectors -> 4. Catch up replication -> 5. Cutover DNS."},
                    {"thought": "Executing Step 1: Provisioning multi-tenant Kubernetes cluster.", "action": "infra_tool.apply_terraform()", "observation": "Cluster provisioned with 8 worker nodes across 3 availability zones."},
                    {"thought": "Executing Step 2: Enabling dual-write Kafka CDC topic.", "action": "kafka_tool.create_cdc_mirror()", "observation": "CDC stream active. 0 dropped events."},
                    {"thought": "Re-evaluating plan: Network throughput is higher than estimated, can parallelize bulk dump.", "action": "planner.update_plan()", "observation": "Plan updated: Spawning 4 parallel partition loaders."}
                ],
                "outcome": "Executed 5-stage migration plan dynamically adjusting partition workers based on real-time network throughput.",
                "efficiency_score": 88
            },
            "Supervisor-Worker: Financial Audit": {
                "pattern": "Supervisor (Hierarchical Multi-Agent)",
                "goal": "Audit Q3 financial disclosures against regulatory SEC 10-Q filings and internal ledger.",
                "steps": [
                    {"thought": "Supervisor delegates document extraction to SEC Filing Specialist.", "action": "delegate_worker(worker='SEC_Analyst')", "observation": "SEC Analyst reports: Stated Q3 Revenue is $42.5M, Net Margin is 18.2%."},
                    {"thought": "Supervisor delegates internal ledger reconciliation to Accounting Specialist.", "action": "delegate_worker(worker='Ledger_Auditor')", "observation": "Ledger Auditor reports: Internal verified revenue is $42.48M (variance of 0.047%)."},
                    {"thought": "Supervisor routes variances to Compliance Critic for threshold verification.", "action": "delegate_worker(worker='Compliance_Critic')", "observation": "Compliance Critic confirms variance is within GAAP materiality tolerance ($20,000 threshold)."}
                ],
                "outcome": "Supervisor consolidated outputs from 3 specialist agents and confirmed regulatory GAAP compliance.",
                "efficiency_score": 95
            },
            "Reflexion & Critic: Secure Code Generation": {
                "pattern": "Self-Correcting Reflection + Critic",
                "goal": "Generate multi-tenant SQL query with Row-Level Security (RLS) enforcement.",
                "steps": [
                    {"thought": "Generator Agent generates initial SQL query string.", "action": "generate_code(query_type='tenant_filter')", "observation": "Initial SQL: 'SELECT * FROM orders WHERE tenant_id = ' + user_input"},
                    {"thought": "Critic Agent inspects generated code against security rubrics.", "action": "critic.evaluate(code, rubric='SQL_Injection_Defense')", "observation": "CRITICAL REJECTION: String concatenation detected. Vulnerable to SQL injection."},
                    {"thought": "Generator Agent reflects on Critic feedback and repairs code.", "action": "reflect_and_regenerate()", "observation": "Patched SQL: 'SELECT * FROM orders WHERE tenant_id = :tenant_id' with parameterized bind variables."},
                    {"thought": "Critic Agent re-evaluates patched implementation.", "action": "critic.evaluate(patched_code)", "observation": "APPROVED: 100% compliance with parameterized query security policies."}
                ],
                "outcome": "Critic detected SQL injection vulnerability; Generator successfully reflected on feedback and patched code.",
                "efficiency_score": 91
            }
        }

    def simulate_pattern(self, pattern_key: str) -> Dict[str, Any]:
        return self.preset_traces.get(pattern_key, self.preset_traces["ReAct: Diagnostic System Troubleshooting"])


# --- LESSON HEADER ---
st.title("38.1 Agent Design Patterns: ReAct, Routers, Supervisors & Reflection")
st.markdown("""
Autonomous AI agents achieve complex enterprise objectives by coordinating cognitive patterns: reasoning, planning,
specialist delegation, and self-reflection. Understanding the mathematical and structural trade-offs of these design patterns
enables engineers to select the optimal cognitive topology for latency, cost, and reliability.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Cognitive Architecture of Autonomous Agent Patterns", divider="rainbow")
st.markdown("""
Modern agent systems move far beyond simple linear pipelines. Depending on the complexity and risk profile of the task,
agents leverage distinct cognitive topologies:
1. **ReAct**: Interleaves reasoning steps with environment tool observations in a tight dynamic feedback loop.
2. **Router**: Classifies intent and routes execution to specialized lightweight agents or deterministic pipelines.
3. **Planner-Executor**: Separates high-level strategic task decomposition from discrete tool execution.
4. **Supervisor**: Organizes specialists hierarchically, managing state transitions and worker delegation.
5. **Reflection / Critic**: Employs an adversarial evaluator loop to inspect, critique, and refine agent actions before finalization.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AgentPatterns_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_input {
            label = "Task Ingestion & Cognitive Router";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            task_in [label="User Objective / Event Stream\\n'Perform audit and patch cluster'", fillcolor="#0F172A"];
            router  [label="Cognitive Router Agent\\nIntent Classification & Complexity Scoring", fillcolor="#1E3A8A"];
        }

        subgraph cluster_patterns {
            label = "Specialized Cognitive Pattern Topologies";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            react_agent [label="ReAct Agent Loop\\nThought -> Tool Action -> Observation", fillcolor="#0369A1"];
            planner_exec [label="Planner-Executor Architecture\\nDecompose -> Step Dispatch -> Dynamic Replan", fillcolor="#7C3AED"];
            supervisor   [label="Supervisor Hierarchical Graph\\nCentral Orchestrator -> Specialist Workers", fillcolor="#047857"];
        }

        subgraph cluster_reflection {
            label = "Evaluative Feedback & Self-Correction";
            style = "filled";
            color = "#1E293B";
            fontcolor = "#38BDF8";
            fontname = "Arial";
            
            critic_node [label="Critic / Evaluator Agent\\nRule Rubrics, Security & Quality Check", fillcolor="#B45309"];
            reflexion   [label="Reflexion Memory & State Repair\\nFeedback Incorporation & Regeneration", fillcolor="#D97706"];
        }

        task_in -> router;
        router  -> react_agent  [label="Single-Domain Interactive"];
        router  -> planner_exec [label="Multi-Stage Long-Horizon"];
        router  -> supervisor   [label="Multi-Domain Cross-Functional"];

        react_agent  -> critic_node [label="Draft Output"];
        planner_exec -> critic_node [label="Milestone State"];
        supervisor   -> critic_node [label="Worker Synthesis"];

        critic_node -> reflexion   [label="Reject with Critique"];
        reflexion   -> supervisor  [label="Self-Corrected Execution"];
        critic_node -> task_in     [label="Approved: Return Final Result"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)


# --- SECTION 2: REACT-STYLE AGENTS ---
st.header("2. ReAct-Style Agents: Reasoning, Action and Observation Interleaving", divider="rainbow")
st.markdown("""
The **ReAct (Reasoning + Acting)** paradigm alternates between generating verbal reasoning traces ("Thoughts") and
issuing structured tool calls ("Actions"), incorporating the resulting external responses ("Observations") back into context.
""")

col_re1, col_re2 = st.columns(2)
with col_re1:
    st.subheader("Core Mechanism & Advantages")
    st.markdown("""
    - **Dynamic Course Correction**: Unlike static chains, the agent updates its internal beliefs after every tool execution.
    - **Reduced Hallucinations**: Grounding reasoning in real environmental feedback prevents the model from generating plausible but fictional facts.
    - **Interpretability**: Engineers can audit every step of the agent's thought trajectory in real time.
    """)

with col_re2:
    st.subheader("Modern Tool-Calling ReAct (LangChain v0.3+)")
    st.code("""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

@tool
def check_server_health(cluster_id: str) -> str:
    \"\"\"Returns CPU, memory, and error rates for a cluster.\"\"\"
    return "CPU: 94%, Mem: 82%, Error Rate: 4.8%"

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm_with_tools = llm.bind_tools([check_server_health])

# ReAct reasoning loop
messages = [
    SystemMessage(content="You are a site reliability engineer. Always reason step-by-step before invoking diagnostic tools."),
    HumanMessage(content="Cluster prod-us-east is reporting alerts. Investigate.")
]
response = llm_with_tools.invoke(messages)
    """, language="python")


# --- SECTION 3: ROUTER AGENTS ---
st.header("3. Router Agents: Intent Classification and Dynamic Specialist Dispatch", divider="rainbow")
st.markdown("""
Not every query requires an expensive multi-step ReAct agent. A **Router Agent** functions as a low-latency gatekeeper,
inspecting the user's intent and routing the request to specialized pipelines, smaller models, or deterministic tools.
""")

col_rt1, col_rt2 = st.columns(2)
with col_rt1:
    st.markdown("### Routing Topologies")
    st.markdown("""
    - **Model Routing**: Directs trivial formatting queries to fast, cost-effective models (e.g., GPT-4o-mini) and complex reasoning tasks to frontier models (e.g., GPT-4o).
    - **Specialist Routing**: Maps domain queries directly to purpose-built agents (e.g., SQL Agent, Billing Agent, Documentation Retriever).
    - **Semantic / Embedding Router**: Uses vector similarity against canonical intent exemplars to achieve sub-10ms routing without LLM token generation.
    """)

with col_rt2:
    st.markdown("### Typed Pydantic Router Schema")
    st.code("""
from enum import Enum
from pydantic import BaseModel, Field

class DestinationRoute(str, Enum):
    SQL_DATABASE = "sql_database"
    KNOWLEDGE_BASE = "knowledge_base"
    HUMAN_ESCALATION = "human_escalation"
    DIRECT_ANSWER = "direct_answer"

class RoutingDecision(BaseModel):
    destination: DestinationRoute
    confidence: float = Field(ge=0.0, le=1.0)
    routing_rationale: str

router_model = llm.with_structured_output(RoutingDecision)
decision = router_model.invoke("Can you show me total revenue by region for last quarter?")
print(decision.destination) # Output: DestinationRoute.SQL_DATABASE
    """, language="python")


# --- SECTION 4: PLANNER-EXECUTOR AGENTS ---
st.header("4. Planner-Executor Agents: Plan-and-Solve and Decomposition Loops", divider="rainbow")
st.markdown("""
ReAct agents make step-by-step decisions greedily, making them vulnerable to getting stuck in local loops on long-horizon tasks.
**Planner-Executor** architectures decouple strategic high-level planning from discrete tactical execution.
""")

col_pe1, col_pe2 = st.columns(2)
with col_pe1:
    st.markdown("""
    ### Two-Phase Execution Cycle
    1. **Planner**: Ingests the objective and decomposes it into an ordered list of atomic sub-tasks with dependency graphs.
    2. **Executor**: Solves each sub-task individually using targeted tools.
    3. **Replanner / Monitor**: Inspects execution outputs against the master plan, pruning completed steps or adding recovery steps when unexpected errors occur.
    """)

with col_pe2:
    st.markdown("### Plan Representation (Pydantic)")
    st.code("""
class SubTask(BaseModel):
    task_id: int
    description: str
    assigned_specialist: str
    dependencies: list[int] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    overall_goal: str
    steps: list[SubTask]
    estimated_duration_sec: int

# Re-planning invocation pattern
def replan(current_plan: ExecutionPlan, completed_step: int, observation: str):
    prompt = f"Plan: {current_plan}. Step {completed_step} produced: {observation}. Update remaining steps."
    return planner_model.invoke(prompt)
    """, language="python")


# --- SECTION 5: SUPERVISOR AGENTS ---
st.header("5. Supervisor Agents: Hierarchical Multi-Agent Orchestration", divider="rainbow")
st.markdown("""
In complex enterprise workflows, individual agents specialize in specific domains (e.g. database querying, code execution, web research).
A **Supervisor Agent** acts as an engineering manager, holding overall responsibility for orchestrating specialists,
managing shared context, and determining when the task is complete.
""")

col_sup1, col_sup2 = st.columns(2)
with col_sup1:
    st.markdown("### Supervisor Delegation Mechanics")
    st.markdown("""
    - **Single Point of Control**: The user communicates exclusively with the Supervisor; specialist workers remain encapsulated.
    - **Context Hygiene**: The Supervisor curates and filters state messages before passing them to workers, preventing context window bloat.
    - **Terminal Routing (`FINISH`)**: The Supervisor explicitly emits a `FINISH` signal when all specialist deliverables satisfy the objective.
    """)

with col_sup2:
    st.markdown("### LangGraph Supervisor Router Pattern")
    st.code("""
from typing import Literal
from pydantic import BaseModel

class SupervisorDecision(BaseModel):
    next_worker: Literal["researcher", "coder", "auditor", "FINISH"]
    instructions_for_worker: str

# Supervisor prompt directs routing
supervisor_prompt = (
    "You are the team supervisor. Given the conversation history, "
    "decide which worker acts next, or choose 'FINISH' if the goal is met."
)
    """, language="python")


# --- SECTION 6: REFLECTION PATTERNS ---
st.header("6. Reflection Patterns: Self-Assessment and Reflexion Loops", divider="rainbow")
st.markdown("""
**Reflection** introduces metacognition into agent workflows. Instead of immediately returning the first generated output,
the agent inspects its own work, evaluates it against quality standards, and iteratively refines its response.
""")

col_ref1, col_ref2 = st.columns(2)
with col_ref1:
    st.markdown("""
    ### Reflexion Framework Architecture
    - **Actor**: Generates initial attempt or code draft.
    - **Evaluator**: Runs unit tests, static analysis, or semantic checks.
    - **Self-Reflector**: Analyzes test failures, generating textual self-critiques stored in episodic memory to prevent repeating the same mistake.
    """)

with col_ref2:
    st.markdown("### Actor-Reflector Loop Implementation")
    st.code("""
class ReflectionCritique(BaseModel):
    is_satisfactory: bool
    flaws_detected: list[str]
    suggested_improvements: str

def critique_draft(task: str, draft: str, reflector_model) -> ReflectionCritique:
    messages = [
        {"role": "system", "content": "Critique this draft rigorously for logic bugs, edge cases, and clarity."},
        {"role": "user", "content": f"Task: {task}\\n\\nDraft:\\n{draft}"}
    ]
    return reflector_model.with_structured_output(ReflectionCritique).invoke(messages)
    """, language="python")


# --- SECTION 7: CRITIC AGENTS ---
st.header("7. Critic Agents: Adversarial and Evaluative Multi-Role Validation", divider="rainbow")
st.markdown("""
Relying on a single model to both generate and evaluate introduces self-confirmation bias.
**Critic Agents** enforce strict role separation: one agent acts as the proactive creator (Generator),
while an independent agent with a distinct persona acts as the rigorous skeptic (Critic/Auditor).
""")

col_cr1, col_cr2 = st.columns(2)
with col_cr1:
    st.markdown("### Generator vs Critic Dynamics")
    crit_table = {
        "Attribute": ["Primary Goal", "Persona Framing", "Temperature", "Prompt Orientation", "Failure Mode"],
        "Generator Agent": ["Creativity & coverage", "Constructive problem solver", "0.2 - 0.7", "Produce complete solution", "Overconfidence / blind spots"],
        "Critic Agent": ["Fault detection & compliance", "Adversarial auditor / skeptic", "0.0", "Find vulnerabilities & edge cases", "Overly pedantic rejections"]
    }
    st.dataframe(pd.DataFrame(crit_table), use_container_width=True, hide_index=True)

with col_cr2:
    st.markdown("### Adversarial Rubric Prompting")
    st.code("""
critic_system_prompt = \"\"\"
You are an adversarial security auditor. Your sole purpose is to find:
1. SQL injection or prompt injection vectors
2. Missing error handling and edge cases
3. Inefficient algorithms or memory leaks
Assume the proposed code is flawed until proven otherwise.
Be rigorous and uncompromising.
\"\"\"
    """, language="python")


# --- SECTION 8: SELF-CORRECTING AGENTS ---
st.header("8. Self-Correcting Agents: Exception Handling and Dynamic State Recovery", divider="rainbow")
st.markdown("""
Production agents operate in volatile external environments where APIs fail, database connections drop,
and models produce malformed outputs. **Self-Correcting Agents** treat exceptions as regular observations,
re-planning dynamically rather than crashing the execution thread.
""")

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    st.markdown("""
    ### Error Recovery Tactics
    - **Parsing Repair**: Catch schema validation errors and feed the traceback directly back into the model prompt to fix the output format.
    - **Tool Fallback Routing**: If a primary API returns a 503 error, the agent autonomously routes the request to a secondary provider.
    - **Context Trimming**: If a request exceeds token limits, the agent compresses or prunes older conversation turns and retries.
    """)

with col_sc2:
    st.markdown("### Self-Correction with Try-Catch in LangGraph")
    st.code("""
def execute_tool_with_recovery(state):
    try:
        result = external_api_call(state["query"])
        return {"result": result, "error": None}
    except Exception as e:
        # Wrap error as an observation for self-correction
        return {
            "error": str(e),
            "messages": [
                {"role": "user", "content": f"Tool failed with error: {str(e)}. Propose alternative approach."}
            ]
        }
    """, language="python")


# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Advanced Multi-Pattern Agent Orchestrator", divider="rainbow")
st.markdown("""
The following production class encapsulates a multi-pattern agent system combining intent routing,
tool execution, and automated critic validation with a maximum retry ceiling.
""")

st.code("""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class SolutionEvaluation(BaseModel):
    is_valid: bool = Field(description="True if output passes all verification rubrics")
    score: float = Field(ge=0.0, le=1.0)
    critique_notes: str = Field(description="Specific feedback or corrections needed")

class ProductionAgentOrchestrator:
    def __init__(self, model_name: str = "gpt-4o", max_retries: int = 3):
        self.generator_llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.critic_llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.max_retries = max_retries

    def execute_with_critic_loop(self, task_description: str, acceptance_criteria: str) -> Dict[str, Any]:
        attempt = 0
        current_critique = "Initial attempt"
        history: List[str] = []

        while attempt < self.max_retries:
            attempt += 1
            # Step 1: Generator drafts solution incorporating any prior critique
            gen_prompt = f"Task: {task_description}\\nCritique from prior turn: {current_critique}\\nGenerate solution."
            solution_draft = str(self.generator_llm.invoke([HumanMessage(content=gen_prompt)]).content)
            
            # Step 2: Critic audits solution against acceptance criteria
            eval_prompt = (
                f"Criteria: {acceptance_criteria}\\n"
                f"Proposed Solution:\\n{solution_draft}\\n"
                f"Evaluate rigorously. If valid, set is_valid=True."
            )
            evaluator = self.critic_llm.with_structured_output(SolutionEvaluation)
            eval_result: SolutionEvaluation = evaluator.invoke([HumanMessage(content=eval_prompt)])
            
            history.append(f"Turn {attempt}: Score {eval_result.score:.2f} - {eval_result.critique_notes}")
            
            if eval_result.is_valid:
                return {
                    "status": "APPROVED",
                    "turns_taken": attempt,
                    "final_output": solution_draft,
                    "history": history
                }
            
            current_critique = eval_result.critique_notes

        return {
            "status": "MAX_RETRIES_EXCEEDED",
            "turns_taken": attempt,
            "final_output": solution_draft,
            "history": history
        }
""", language="python")


# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Agent Design Pattern Simulator", divider="rainbow")
st.markdown("""
Experiment with different agent cognitive topologies. Inspect reasoning traces, tool dispatches,
critic evaluations, and dynamic self-correction loops.
""")

sim = AgentDesignPatternSimulator()

col_bench_cfg, col_bench_res = st.columns([1, 1])

with col_bench_cfg:
    st.subheader("Select Cognitive Pattern Scenario")
    selected_scenario = st.selectbox(
        "Choose Agent Pattern Scenario:",
        list(sim.preset_traces.keys()),
        index=0
    )
    
    scenario_info = sim.preset_traces[selected_scenario]
    st.markdown(f"**Pattern Class**: `{scenario_info['pattern']}`")
    st.text_area("Scenario Objective:", value=scenario_info["goal"], height=80, disabled=True)
    
    run_sim_btn = st.button("🚀 Execute Pattern Trajectory", use_container_width=True)

with col_bench_res:
    st.subheader("Agent Execution Telemetry & Step Trace")
    if run_sim_btn:
        res = sim.simulate_pattern(selected_scenario)
        
        st.markdown(
            f"<div class='pattern-card'>"
            f"<b>Active Pattern:</b> <span class='badge-react'>{res['pattern']}</span><br><br>"
            f"<b>Efficiency Rating:</b> {res['efficiency_score']}%<br>"
            f"<b>Total Cognitive Steps:</b> {len(res['steps'])}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("### Step-by-Step Trajectory")
        for i, step in enumerate(res["steps"], 1):
            st.markdown(
                f"<div class='step-box'>"
                f"<b>Step {i} Thought:</b> {step['thought']}<br>"
                f"<b>Action:</b> <code>{step['action']}</code><br>"
                f"<b>Observation:</b> {step['observation']}"
                f"</div>",
                unsafe_allow_html=True
            )
            
        st.success(f"Final Outcome: {res['outcome']}")
    else:
        st.info("Select a design pattern scenario and click Execute to observe the reasoning trajectory.")


# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is the fundamental architectural trade-off between ReAct and Planner-Executor agents?"):
    st.markdown("""
    **Answer**: ReAct agents make greedy, step-by-step decisions, which makes them highly responsive to unexpected tool observations
    but vulnerable to wandering or getting stuck in loops on long-horizon tasks. Planner-Executor agents generate a holistic multi-step plan upfront,
    making them structurally sound on complex tasks, but require a replanning mechanism to handle unexpected tool failures gracefully.
    """)

with st.expander("Q2: Why should Generator and Critic roles be assigned to separate model calls rather than one unified prompt?"):
    st.markdown("""
    **Answer**: When a single prompt is asked to generate and critique simultaneously, the attention mechanism exhibits confirmation bias,
    tending to justify earlier token choices. Separating roles into independent calls—often with distinct system prompts and temperature settings
    (higher temperature for the Generator, temperature 0.0 for the Critic)—ensures objective adversarial validation.
    """)

with st.expander("Q3: How does a Supervisor Agent prevent context window exhaustion in multi-agent workflows?"):
    st.markdown("""
    **Answer**: The Supervisor acts as a context firewall. Instead of maintaining a shared, ever-growing message history across all workers,
    the Supervisor isolates worker conversations into independent sub-graphs and only incorporates concise worker summaries into the top-level state.
    """)

with st.expander("Q4: What is the role of 'Reflexion' memory in self-correcting agents?"):
    st.markdown("""
    **Answer**: Reflexion augments agents with an episodic memory buffer of past failures and self-critiques.
    When a tool returns an error, the agent generates an explicit textual critique explaining why the action failed.
    This critique is prepended to future attempts, steering the model away from repeating identical mistakes.
    """)

with st.expander("Q5: When is a semantic vector router preferred over an LLM-based router?"):
    st.markdown("""
    **Answer**: A semantic vector router computes cosine similarity against pre-embedded intent exemplars without generating tokens.
    It executes in under 10ms with zero LLM API cost, making it ideal for high-throughput, low-latency classification across fixed intent categories.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Unbounded ReAct Infinite Loops)**: Failing to enforce `max_iterations` on ReAct loops. An agent facing ambiguous tool outputs will cycle endlessly, draining API quotas.")
    st.warning("**Trap 2 (Blind Planning Without Replanning)**: Executing a 10-step plan sequentially without checking intermediate outputs. An early failure cascades into garbage downstream results.")
with col_trap2:
    st.error("**Trap 3 (Shared Unbounded Multi-Agent State)**: Passing the entire conversation history to all multi-agent workers, inflating token costs by O(N^2) and causing context window limits to fail.")
    st.warning("**Trap 4 (Vague Critic Rubrics)**: Instructing a Critic to 'review for quality' without specific pass/fail criteria, leading to superficial approvals or endless rejection loops.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What does ReAct stand for?")
        st.caption("Flip for answer:")
        st.info("**Back**: Reasoning and Acting (interleaving Thought, Action, and Observation).")
    with st.container(border=True):
        st.markdown("**Front**: What is the primary benefit of Planner-Executor over pure ReAct?")
        st.caption("Flip for answer:")
        st.info("**Back**: Decouples strategic planning from tactical tool execution to prevent greedy local-loop traps.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What does a Supervisor Agent emit to signal task completion?")
        st.caption("Flip for answer:")
        st.info("**Back**: The terminal `FINISH` action/route to end orchestration.")
    with st.container(border=True):
        st.markdown("**Front**: How does Reflexion prevent repeating errors?")
        st.caption("Flip for answer:")
        st.info("**Back**: Stores textual self-critiques of past failures in working memory to guide future attempts.")

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the mechanics and trade-offs of ReAct vs Planner-Executor agents.", value=True)
c2 = st.checkbox("I can implement typed Router agents using Pydantic schemas.", value=True)
c3 = st.checkbox("I know how to design hierarchical Supervisor architectures with terminal routing.", value=True)
c4 = st.checkbox("I understand how to separate Generator and Critic roles to avoid confirmation bias.", value=True)
c5 = st.checkbox("I can build self-correcting agent loops with automated retry ceilings.", value=True)
c6 = st.checkbox("I know how to prevent context window explosion in multi-agent workflows.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 38.2: Agent Memory.")
