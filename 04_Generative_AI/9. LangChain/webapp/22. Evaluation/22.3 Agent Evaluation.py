import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.3 Agent Evaluation - LangChain Master Curriculum",
    page_icon="🤖",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class AgentTrajectoryEvaluationEngine:
    """
    Enterprise Agent Trajectory & Multi-Step Reasoning Evaluation Engine.
    Simulates evaluation across the 6 core agent evaluation pillars:
    1. Tool Selection Precision (Selecting the optimal tool vs distractors)
    2. Tool-Call Correctness (Valid JSON schema parameters and arguments)
    3. Task Completion Rate (End-to-end goal resolution)
    4. Trajectory Step Efficiency (Step count, loops, redundant hops)
    5. Error Handling & Self-Correction (Recovery from simulated tool crashes)
    6. Multi-Step Reasoning Coherence (Intermediate thought verification)
    """
    def __init__(self, agent_name: str = "customer-ops-agent"):
        self.agent_name = agent_name
        self.trajectory_history: List[Dict[str, Any]] = []

    def evaluate_trajectory(
        self,
        task_prompt: str,
        expected_tools: List[str],
        actual_trajectory: List[Dict[str, Any]],
        final_answer: str,
        max_allowed_steps: int = 5,
        simulate_tool_crash: bool = False
    ) -> Dict[str, Any]:
        """
        Audits an agent's multi-step execution trajectory against production criteria.
        """
        # 1. Tool Selection Precision
        executed_tool_names = [step["tool"] for step in actual_trajectory if step.get("tool")]
        correct_tools_chosen = [t for t in executed_tool_names if t in expected_tools]
        tool_selection_precision = len(correct_tools_chosen) / max(1, len(executed_tool_names))

        # 2. Tool-Call Argument Correctness
        schema_passes = sum(1 for step in actual_trajectory if step.get("schema_valid", True))
        tool_call_correctness = schema_passes / max(1, len(actual_trajectory))

        # 3. Trajectory Step Efficiency
        total_steps = len(actual_trajectory)
        step_efficiency_score = max(0.0, 1.0 - max(0, total_steps - len(expected_tools)) * 0.15)
        has_redundant_loops = len(executed_tool_names) != len(set(executed_tool_names)) and not simulate_tool_crash

        # 4. Error Handling & Recovery
        if simulate_tool_crash:
            # Check if trajectory contains error recovery step
            recovered = any(step.get("is_recovery", False) for step in actual_trajectory)
            error_recovery_score = 1.0 if recovered else 0.0
        else:
            error_recovery_score = 1.0

        # 5. Task Completion
        task_completed = (len(correct_tools_chosen) == len(expected_tools)) and (len(final_answer.strip()) > 20) and not has_redundant_loops
        task_completion_score = 1.0 if task_completed else 0.3

        # 6. Composite Score (Weighted Synthesis)
        composite_score = round(
            (0.25 * task_completion_score) +
            (0.25 * tool_selection_precision) +
            (0.20 * tool_call_correctness) +
            (0.15 * step_efficiency_score) +
            (0.15 * error_recovery_score),
            3
        )

        passed = composite_score >= 0.80 and task_completed

        report = {
            "task_prompt": task_prompt,
            "total_steps": total_steps,
            "max_allowed_steps": max_allowed_steps,
            "composite_score": composite_score,
            "passed": passed,
            "verdict": "CERTIFIED_AGENT_TRAJECTORY" if passed else "FAILED_AGENT_QUALITY_GATE",
            "metrics": {
                "task_completion": task_completion_score,
                "tool_selection_precision": tool_selection_precision,
                "tool_call_correctness": tool_call_correctness,
                "step_efficiency": step_efficiency_score,
                "error_recovery": error_recovery_score
            },
            "trajectory_steps": actual_trajectory,
            "final_answer": final_answer
        }
        self.trajectory_history.append(report)
        return report

# Initialize engine in session state
if "agent_eval_engine" not in st.session_state:
    st.session_state.agent_eval_engine = AgentTrajectoryEvaluationEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("22.3 Agent Evaluation")
st.caption("Chapter 22: Evaluation | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Autonomous agents built with LangChain and LangGraph do not simply transform text; they execute multi-step decision loops: 
analyzing user queries, selecting tools from an inventory, generating structured API parameters, interpreting observation results, 
recovering from tool crashes, and deciding when the goal has been satisfied. 
Evaluating agents requires looking inside the **execution trajectory** rather than just judging the final string.

In this lesson, you will master:
1. Tool selection precision and avoiding irrelevant or dangerous tool execution
2. Tool-call parameter correctness and JSON schema validation
3. End-to-end task completion rates and stopping condition verification
4. Trajectory efficiency: detecting redundant hops, loops, and decision deadlocks
5. Automated error recovery and multi-step reasoning plan validation
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Agent Trajectory and Multi-Step Execution Evaluation Architecture", divider="rainbow")

st.markdown("""
Agent evaluation monitors the agent's thought-action-observation loop. 
A successful agent must not only reach the right final answer; it must take the optimal path, call tools with valid arguments, 
and handle intermediate API exceptions gracefully without spinning in infinite loops.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AgentTrajectoryEvaluation {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_agent_loop {
            label = "1. Agent Trajectory Execution Stream";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            STEP1 [label="Step 1: Reason & Select Tool\\nTool: 'crm_customer_lookup'", fillcolor="#1E3A8A", color="#3B82F6"];
            STEP2 [label="Step 2: Generate Arguments\\nArgs: {'customer_id': 'cust_982'}", fillcolor="#1E3A8A", color="#3B82F6"];
            STEP3 [label="Step 3: Execute & Observe\\nResult: {'tier': 'enterprise', 'status': 'active'}", fillcolor="#1E3A8A", color="#3B82F6"];
            STEP4 [label="Step 4: Final Synthesis\\nGenerates User Answer", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_trajectory_evaluators {
            label = "2. Trajectory Evaluation Dimensions";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F59E0B";
            fontsize = 12;

            E_TOOL [label="Tool Choice Evaluator\\n(Did it select CRM vs Calculator?)", fillcolor="#7C2D12", color="#EA580C"];
            E_SCHEMA [label="Tool Schema Evaluator\\n(Did args match Pydantic spec?)", fillcolor="#7C2D12", color="#EA580C"];
            E_EFF [label="Trajectory Efficiency\\n(Shortest path vs infinite loops)", fillcolor="#7C2D12", color="#EA580C"];
            E_RECOV [label="Error Recovery Auditor\\n(Did it self-heal from API timeouts?)", fillcolor="#7C2D12", color="#EA580C"];
            E_GOAL [label="Task Completion Auditor\\n(Did it satisfy user prompt?)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_verdict {
            label = "3. Automated Certification Verdict";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            PASS [label="Certified Trajectory\\n(Optimal path, zero schema errors)", fillcolor="#065F46", color="#059669"];
            FAIL [label="Failed Trajectory\\n(Redundant loop or wrong tool called)", fillcolor="#881337", color="#F43F5E"];
        }

        STEP1 -> E_TOOL [label="Evaluates Tool Choice"];
        STEP2 -> E_SCHEMA [label="Validates Parameters"];
        STEP1 -> STEP2 -> STEP3 -> STEP4;
        STEP3 -> E_RECOV [label="Inspects Exceptions"];
        STEP4 -> E_GOAL [label="Validates Output"];

        STEP1 -> E_EFF [label="Count Steps"];
        STEP3 -> E_EFF [label="Count Steps"];

        E_TOOL -> PASS [label="Passes Matrix"];
        E_SCHEMA -> PASS;
        E_EFF -> PASS;
        E_GOAL -> PASS;

        E_EFF -> FAIL [label="Loop Detected"];
        E_TOOL -> FAIL [label="Wrong Tool"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Agent Evaluation
1. **Never Evaluate an Agent Solely on Its Final Answer**: An agent might guess the right number on pure chance after making 15 unnecessary, costly tool calls. Evaluating the trajectory ensures the agent took the shortest path without burning tokens.
2. **Schema Invariant Checking Must Be Zero-Cost**: Validating tool call arguments against JSON schemas should be handled deterministically via Pydantic validators before invoking LLM judges.
3. **Hard Recursion Ceilings Are Mandatory**: Trajectory evaluation must assert that an agent terminates within a defined step budget (`steps <= max_allowed_steps`), failing any run that exhibits repetitive cycle loops.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOOL SELECTION
# ==============================================================================
st.header("2. Tool Selection (Tool Choice Precision, Negative Tool Avoidance and Intent Mapping)", divider="rainbow")

st.markdown("""
In enterprise environments, an agent is often equipped with dozens of tools: database lookups, calculators, 
CRM APIs, payment gateways, and web search engines. Tool selection evaluation assesses whether the agent 
invoked the exact tool intended for the task, while avoiding irrelevant distractors or dangerous tools.
""")

st.markdown("""
### The Two Failure Modes of Tool Selection
* **False Tool Invocation (Over-Triggering)**: The agent invokes an external tool when the answer could have been resolved directly from its system prompt or context, incurring unnecessary latency and API cost.
* **Wrong Tool Selection (Misrouting)**: When asked to calculate an account balance, the agent calls `web_search` instead of `sql_database_tool`.
* **Negative Tool Avoidance**: Verifying that destructive tools (`delete_user_account`, `issue_refund`) are NEVER called on read-only queries.
""")

tool_sel_df = pd.DataFrame([
    {"Evaluation Metric": "Tool Selection Precision", "Definition": "(Correct Tools Selected) / (Total Tools Invoked)", "Target SLA": "1.00 (Zero unnecessary tools)"},
    {"Evaluation Metric": "Tool Selection Recall", "Definition": "(Required Tools Invoked) / (Total Tools Needed)", "Target SLA": "1.00 (All necessary steps executed)"},
    {"Evaluation Metric": "Destructive Tool Violation Rate", "Definition": "Frequency of dangerous tool invocation on benign prompts", "Target SLA": "0.00% (Strict zero tolerance)"}
])
st.dataframe(tool_sel_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern LangChain stacks use `bind_tools()` with strict tool descriptions. 
Evaluation benchmarks measure tool selection accuracy across an inventory of 20 candidate tools, ensuring the model's 
intent routing remains sharp as tool catalogs expand.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOOL-CALL CORRECTNESS
# ==============================================================================
st.header("3. Tool-Call Correctness (Schema Conformance, Parameter Accuracy and Argument Fidelity)", divider="rainbow")

st.markdown("""
Selecting the right tool is useless if the agent generates malformed arguments. 
Tool-call correctness evaluates whether the generated parameters match the tool's JSON schema types, required keys, 
and semantic constraints.
""")

st.markdown("""
### The 3 Layers of Parameter Validation
1. **Syntactic Conformance**: Is the argument payload valid JSON?
2. **Schema Type Conformance**: Does an integer parameter receive an integer rather than a string? Are all mandatory fields present?
3. **Semantic Fidelity**: Did the agent extract the exact parameter from the user query? (e.g., query states *"Lookup user ID 982"*; did the agent pass `customer_id=982` or `customer_id=123`?)
""")

st.code('''
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CustomerLookupSchema(BaseModel):
    customer_id: str = Field(description="Unique alphanumeric customer ID.")
    include_billing_history: bool = Field(default=False, description="Flag to fetch invoices.")

@tool(args_schema=CustomerLookupSchema)
def crm_lookup_tool(customer_id: str, include_billing_history: bool = False):
    """Look up customer records from enterprise CRM."""
    return {"status": "active", "tier": "enterprise"}

# Programmatic evaluation of tool arguments
def evaluate_tool_call_args(tool_call: dict, expected_args: dict) -> float:
    args = tool_call.get("args", {})
    correct_keys = sum(1 for k, v in expected_args.items() if args.get(k) == v)
    return correct_keys / max(1, len(expected_args))
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
With native tool calling in modern chat models (`ChatOpenAI`, `ChatAnthropic`), models output structured JSON directly. 
LangChain verifies arguments against Pydantic schemas, surfacing schema validation errors directly to the evaluator.
""")

st.divider()

# ==============================================================================
# SECTION 4: TASK COMPLETION
# ==============================================================================
st.header("4. Task Completion (End-to-End Success Rate, Goal Satisfaction and Stopping Criteria)", divider="rainbow")

st.markdown("""
Task completion measures whether the agent successfully satisfied the end-user's objective. 
In multi-step autonomous workflows, an agent can execute several successful tool calls but stop prematurely 
without synthesizing a final resolution, or generate an answer before fetching required information.
""")

st.markdown("""
### Evaluating Goal Satisfaction
* **End-to-End Task Success Rate**: The percentage of test queries where the agent achieved the requested business outcome (e.g., booked the flight, retrieved the statement, created the ticket).
* **Proper Termination Criteria**: Verifying that the agent terminated cleanly by emitting a final message rather than crashing into recursion limits or timeout bounds.
* **Negative Goal Satisfaction**: Verifying that when a user asks for an unauthorized action, the agent cleanly refuses and terminates.
""")

st.code('''
# LangChain Task Completion Evaluator Pattern
from langchain.evaluation import load_evaluator

task_evaluator = load_evaluator("labeled_criteria", criteria="correctness")
eval_result = task_evaluator.evaluate_strings(
    prediction=agent_final_output,
    input=user_objective,
    reference=golden_resolution
)
print(f"Goal Completion Score: {eval_result['score']}, Rationale: {eval_result['reasoning']}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams track task completion rates across release versions. 
A candidate agent version must achieve >= 95% task completion on golden benchmarks before canary rollout.
""")

st.divider()

# ==============================================================================
# SECTION 5: AGENT TRAJECTORY
# ==============================================================================
st.header("5. Agent Trajectory (Step Efficiency, Redundant Loop Avoidance and Shortest-Path Metrics)", divider="rainbow")

st.markdown("""
The execution trajectory is the ordered sequence of thoughts, tool calls, and observations that an agent takes to resolve a task. 
Trajectory evaluation assesses how efficiently the agent navigated the decision space.
""")

st.markdown("""
### Trajectory Pathology Anti-Patterns
* **Redundant Loop Cycles**: Calling the same tool repeatedly with identical parameters because the agent failed to interpret the observation.
* **Detour Hops**: Taking 6 tool steps to resolve a question that required 2 tool steps, burning tokens and inflating user latency.
* **Deadlocks**: The agent gets confused by tool output and enters an infinite reasoning loop until stopped by hard execution limits.
""")

trajectory_metrics_table = pd.DataFrame([
    {"Trajectory Metric": "Step Overhead Ratio", "Formula / Meaning": "(Actual Steps) / (Optimal Golden Steps)", "Target SLA": "<= 1.20 (Minimal detours)"},
    {"Trajectory Metric": "Loop Detection", "Formula / Meaning": "Presence of identical tool calls in sequence", "Target SLA": "0.00 (Zero duplicate loops)"},
    {"Trajectory Metric": "Token Trajectory Burn", "Formula / Meaning": "Total cumulative tokens across all reasoning steps", "Target SLA": "< 4,000 tokens per task"}
])
st.dataframe(trajectory_metrics_table, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangGraph workflows make trajectories explicit as state transitions. 
Trajectory evaluators analyze the list of visited nodes and edge transitions recorded in the LangSmith trace, 
verifying that the graph traversed the optimal execution path.
""")

st.divider()

# ==============================================================================
# SECTION 6: ERROR HANDLING
# ==============================================================================
st.header("6. Error Handling (Tool Crash Resilience, Self-Correction and Recovery Loops)", divider="rainbow")

st.markdown("""
In the real world, tools fail: databases time out, external payment APIs return 500 errors, and search engines 
return empty results. A brittle agent crashes immediately; a resilient agent observes the error message, 
adjusts its strategy, and tries an alternative path.
""")

st.markdown("""
### Evaluating Agent Self-Correction
* **Tool Crash Injection**: In benchmark evaluations, test suites deliberately simulate tool exceptions (e.g., returning `HTTP 503 Service Unavailable`).
* **Self-Healing Verification**: Does the agent inspect the error and try a fallback tool?
* **Graceful Degradation**: If all fallbacks fail, does the agent explain the situation politely to the user, or does it crash with an unhandled Python stack trace?
""")

st.code('''
# Self-Correction Verification Pattern in Trajectory
def verify_agent_self_correction(trajectory: List[dict]) -> bool:
    """
    Asserts that the agent attempted a recovery step following a tool error.
    """
    for i, step in enumerate(trajectory[:-1]):
        if step.get("observation_error"):
            # Check if next step tried a fallback or adapted parameters
            next_step = trajectory[i + 1]
            if next_step.get("tool") != step.get("tool") or next_step.get("is_recovery"):
                return True
    return False
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise chaos engineering suites inject synthetic latency and failures into tool endpoints during nightly CI runs. 
Agents that fail to self-correct within 2 attempts are blocked from production.
""")

st.divider()

# ==============================================================================
# SECTION 7: MULTI-STEP REASONING EVALUATION
# ==============================================================================
st.header("7. Multi-Step Reasoning Evaluation (Intermediate Thought Coherence and Plan Validation)", divider="rainbow")

st.markdown("""
Complex agents perform multi-step planning (e.g., ReAct, Plan-and-Solve) before invoking tools. 
Evaluating reasoning assesses whether the agent's intermediate "thoughts" are logically coherent, 
mathematically sound, and aligned with the ultimate user goal.
""")

st.markdown("""
### Evaluating Intermediate Reasoning
* **Thought Coherence**: Does the agent's explanation justify its choice of the subsequent tool?
* **Hallucinated Observations**: Does the agent hallucinate facts in its reasoning that were NOT present in the tool's observation payload?
* **Planning Validity**: Did the agent break down a complex multi-part request into the correct sequence of atomic actions?
""")

st.code('''
# Evaluating Reasoning Coherence via LLM Judge
REASONING_RUBRIC_PROMPT = """You are auditing an AI agent's internal reasoning step.
User Query: {query}
Agent's Thought: {thought}
Selected Tool: {tool_name}
Observation Received: {observation}

Evaluate:
1. Is the thought logically sound? (1 or 0)
2. Does the thought accurately reflect the observation without hallucination? (1 or 0)
3. Did the thought justify the next action? (1 or 0)
"""
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production LangGraph agents log intermediate thoughts as structured state updates. 
Evaluator daemons audit these thought states to detect reasoning drift before catastrophic decisions occur.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Agent Trajectory Evaluation Engine (AgentTrajectoryEvaluationEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the trajectory evaluation engine. 
It analyzes tool selection precision, schema argument validity, step counts, error recovery loops, and task completion.
""")

st.code('''
class AgentTrajectoryEvaluationEngine:
    """
    Enterprise Agent Trajectory & Multi-Step Reasoning Evaluation Engine.
    Simulates evaluation across the 6 core agent evaluation pillars:
    1. Tool Selection Precision (Selecting the optimal tool vs distractors)
    2. Tool-Call Correctness (Valid JSON schema parameters and arguments)
    3. Task Completion Rate (End-to-end goal resolution)
    4. Trajectory Step Efficiency (Step count, loops, redundant hops)
    5. Error Handling & Self-Correction (Recovery from simulated tool crashes)
    6. Multi-Step Reasoning Coherence (Intermediate thought verification)
    """
    def __init__(self, agent_name: str = "customer-ops-agent"):
        self.agent_name = agent_name
        self.trajectory_history: List[Dict[str, Any]] = []

    def evaluate_trajectory(
        self,
        task_prompt: str,
        expected_tools: List[str],
        actual_trajectory: List[Dict[str, Any]],
        final_answer: str,
        max_allowed_steps: int = 5,
        simulate_tool_crash: bool = False
    ) -> Dict[str, Any]:
        """
        Audits an agent's multi-step execution trajectory against production criteria.
        """
        # 1. Tool Selection Precision
        executed_tool_names = [step["tool"] for step in actual_trajectory if step.get("tool")]
        correct_tools_chosen = [t for t in executed_tool_names if t in expected_tools]
        tool_selection_precision = len(correct_tools_chosen) / max(1, len(executed_tool_names))

        # 2. Tool-Call Argument Correctness
        schema_passes = sum(1 for step in actual_trajectory if step.get("schema_valid", True))
        tool_call_correctness = schema_passes / max(1, len(actual_trajectory))

        # 3. Trajectory Step Efficiency
        total_steps = len(actual_trajectory)
        step_efficiency_score = max(0.0, 1.0 - max(0, total_steps - len(expected_tools)) * 0.15)
        has_redundant_loops = len(executed_tool_names) != len(set(executed_tool_names)) and not simulate_tool_crash

        # 4. Error Handling & Recovery
        if simulate_tool_crash:
            recovered = any(step.get("is_recovery", False) for step in actual_trajectory)
            error_recovery_score = 1.0 if recovered else 0.0
        else:
            error_recovery_score = 1.0

        # 5. Task Completion
        task_completed = (len(correct_tools_chosen) == len(expected_tools)) and (len(final_answer.strip()) > 20) and not has_redundant_loops
        task_completion_score = 1.0 if task_completed else 0.3

        composite_score = round(
            (0.25 * task_completion_score) +
            (0.25 * tool_selection_precision) +
            (0.20 * tool_call_correctness) +
            (0.15 * step_efficiency_score) +
            (0.15 * error_recovery_score),
            3
        )

        passed = composite_score >= 0.80 and task_completed

        report = {
            "task_prompt": task_prompt,
            "total_steps": total_steps,
            "max_allowed_steps": max_allowed_steps,
            "composite_score": composite_score,
            "passed": passed,
            "verdict": "CERTIFIED_AGENT_TRAJECTORY" if passed else "FAILED_AGENT_QUALITY_GATE",
            "metrics": {
                "task_completion": task_completion_score,
                "tool_selection_precision": tool_selection_precision,
                "tool_call_correctness": tool_call_correctness,
                "step_efficiency": step_efficiency_score,
                "error_recovery": error_recovery_score
            },
            "trajectory_steps": actual_trajectory,
            "final_answer": final_answer
        }
        self.trajectory_history.append(report)
        return report
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Agent Trajectory Simulator and Tool-Calling Auditor", divider="rainbow")

st.markdown("""
Audit agent execution trajectories across multiple operational scenarios. 
Test how redundant loops, tool schema errors, and simulated tool crashes impact trajectory scores.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Agent Trajectory Scenarios")
    traj_scenario = st.selectbox(
        "Execution Scenario Preset",
        [
            "Optimal Trajectory (Direct Tool Path)",
            "Redundant Loop (Calling Same Tool Repeatedly)",
            "Tool Crash & Successful Self-Correction",
            "Schema Argument Error (Malformed JSON)"
        ],
        index=0
    )

    p_prompt = "Look up customer account 982 and check active subscription tier."
    exp_tools = ["crm_customer_lookup"]

    if "Optimal" in traj_scenario:
        actual_steps = [
            {"step": 1, "thought": "Look up customer record.", "tool": "crm_customer_lookup", "args": {"id": "982"}, "schema_valid": True, "observation": "Tier: Enterprise"}
        ]
        f_ans = "Customer 982 has an active Enterprise subscription tier."
        crash_sim = False
    elif "Redundant" in traj_scenario:
        actual_steps = [
            {"step": 1, "thought": "Look up record.", "tool": "crm_customer_lookup", "args": {"id": "982"}, "schema_valid": True, "observation": "Tier: Enterprise"},
            {"step": 2, "thought": "Verify record again.", "tool": "crm_customer_lookup", "args": {"id": "982"}, "schema_valid": True, "observation": "Tier: Enterprise"},
            {"step": 3, "thought": "Check once more.", "tool": "crm_customer_lookup", "args": {"id": "982"}, "schema_valid": True, "observation": "Tier: Enterprise"}
        ]
        f_ans = "Customer 982 has Enterprise tier."
        crash_sim = False
    elif "Crash" in traj_scenario:
        actual_steps = [
            {"step": 1, "thought": "Look up record in primary CRM.", "tool": "crm_customer_lookup", "args": {"id": "982"}, "schema_valid": True, "observation_error": True, "observation": "HTTP 503 Timeout"},
            {"step": 2, "thought": "Primary CRM timed out. Trying read-replica fallback.", "tool": "crm_replica_lookup", "is_recovery": True, "args": {"id": "982"}, "schema_valid": True, "observation": "Tier: Enterprise"}
        ]
        exp_tools = ["crm_customer_lookup", "crm_replica_lookup"]
        f_ans = "Customer 982 has an Enterprise subscription (retrieved via fallback replica)."
        crash_sim = True
    else:  # Schema Error
        actual_steps = [
            {"step": 1, "thought": "Look up record.", "tool": "crm_customer_lookup", "args": {"wrong_field": 982}, "schema_valid": False, "observation": "Pydantic ValidationError"}
        ]
        f_ans = "Could not fetch customer details."
        crash_sim = False

    st.markdown(f"**Task Prompt**: `{p_prompt}`")
    st.markdown(f"**Expected Tools**: `{exp_tools}`")
    st.markdown(f"**Steps Taken**: `{len(actual_steps)}`")

with wb_col2:
    st.markdown("### Live Trajectory Audit Report")
    traj_report = st.session_state.agent_eval_engine.evaluate_trajectory(
        task_prompt=p_prompt,
        expected_tools=exp_tools,
        actual_trajectory=actual_steps,
        final_answer=f_ans,
        max_allowed_steps=5,
        simulate_tool_crash=crash_sim
    )

    if traj_report["passed"]:
        st.success(f"**VERDICT: {traj_report['verdict']}** — Trajectory score {traj_report['composite_score']:.3f} meets enterprise qualification.")
    else:
        st.error(f"**VERDICT: {traj_report['verdict']}** — Trajectory score {traj_report['composite_score']:.3f} breached quality threshold.")

    # High-level Metrics Row
    m = traj_report["metrics"]
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Tool Selection Precision", f"{m['tool_selection_precision']*100:.0f}%")
    with mc2:
        st.metric("Schema Correctness", f"{m['tool_call_correctness']*100:.0f}%")
    with mc3:
        st.metric("Step Efficiency", f"{m['step_efficiency']*100:.0f}%")

    # Trajectory Step Table
    st.markdown("#### Audited Trajectory Sequence")
    st.dataframe(pd.DataFrame(traj_report["trajectory_steps"]), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

with st.expander("Q1: Why is evaluating an agent's trajectory more informative than evaluating its final output?"):
    st.markdown("""
    **Answer**: An agent can arrive at a correct final answer through a deeply pathological execution path: 
    making 15 redundant tool calls, executing irrelevant external APIs, or entering retry loops that burn thousands 
    of unnecessary tokens and add 10 seconds of latency. Trajectory evaluation audits tool choice precision, 
    schema validity, step efficiency, and error recovery, ensuring the agent operates via the optimal path.
    """)

with st.expander("Q2: How do you measure Tool Selection Precision in enterprise agent systems?"):
    st.markdown("""
    **Answer**: Tool selection precision is calculated as: `(Number of Correct Tools Invoked) / (Total Tools Invoked)`. 
    In benchmarks with 20 candidate tools, precision measures whether the agent selected only the required tools 
    without executing distractors or over-triggering external services for questions that could be resolved directly.
    """)

with st.expander("Q3: What causes agent 'Loop Cycles', and how are they detected during evaluation?"):
    st.markdown("""
    **Answer**: Loop cycles occur when an agent receives an observation that it fails to understand or that does not 
    satisfy its stopping condition, causing it to call the identical tool with identical parameters repeatedly. 
    Trajectory evaluators detect loops by hashing consecutive tool calls; if identical hashes appear in sequence, 
    the run is flagged as a loop failure and terminated.
    """)

with st.expander("Q4: How do you evaluate an agent's self-correction and error resilience?"):
    st.markdown("""
    **Answer**: Automated chaos testing injects synthetic exceptions (e.g., HTTP 500, empty search results, malformed JSON) 
    into tool observation responses. The evaluation pipeline checks whether the subsequent step in the trajectory 
    adapts its strategy (trying an alternative tool, adjusting parameters, or informing the user) rather than crashing 
    with an unhandled exception.
    """)

with st.expander("Q5: What is Step Overhead Ratio in agent trajectory benchmarking?"):
    st.markdown("""
    **Answer**: Step overhead ratio is defined as: `(Actual Trajectory Steps Taken) / (Optimal Golden Steps Required)`. 
    A ratio of 1.0 indicates that the agent took the shortest possible path. Ratios above 1.5 indicate significant 
    meandering, redundant tool hops, or inefficient reasoning that inflates production costs and latency.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_at1, col_at2 = st.columns(2)
with col_at1:
    st.error("**Trap 1 (Unbounded Execution Loops)**: Running agents without hard recursion limits (`recursion_limit=15` in StateGraph). A single reasoning deadlock can execute 100 consecutive tool calls, burning hundreds of dollars in minutes.")
    st.warning("**Trap 2 (Ignoring Schema Validation Errors)**: Allowing tool calls to fail at the database level rather than validating parameters client-side. Always bind Pydantic schemas via `bind_tools()` to catch malformed arguments before network execution.")
with col_at2:
    st.error("**Trap 3 (The Guessing Fallback)**: When a tool fails, allowing the agent to guess the answer from training memory. This transforms a simple API timeout into a severe customer-facing hallucination. Resilient agents must report the tool outage explicitly.")
    st.warning("**Trap 4 (Evaluating Without Destructive Tool Probes)**: Testing agents only with read-only tools. Production suites must verify that dangerous tools (`delete_account`, `transfer_funds`) are never called on read-only queries.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 6 pillars of Agent Evaluation?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Tool Selection, 2. Tool-Call Correctness, 3. Task Completion, 4. Trajectory Efficiency, 5. Error Recovery, 6. Multi-Step Reasoning Coherence.")

    with st.container(border=True):
        st.markdown("**Front**: What is Tool Selection Recall?")
        st.caption("Flip for answer:")
        st.info("**Back**: `(Required Tools Invoked) / (Total Required Tools)`. Verifies that the agent did not skip necessary steps or information retrieval.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does LangGraph facilitate trajectory evaluation?")
        st.caption("Flip for answer:")
        st.info("**Back**: By modeling execution as explicit state transitions across graph nodes; evaluators inspect the node history directly in checkpoints or LangSmith traces.")

    with st.container(border=True):
        st.markdown("**Front**: What is the 'Step Overhead Ratio' target SLA for production agents?")
        st.caption("Flip for answer:")
        st.info("**Back**: `<= 1.20`. Agents should take no more than 20% more steps than the optimal golden trajectory.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Agent Evaluation in LangChain:")

c1 = st.checkbox("I know how to evaluate tool selection precision against distractor tool inventories.", value=True)
c2 = st.checkbox("I can validate tool-call parameter correctness against strict Pydantic JSON schemas.", value=True)
c3 = st.checkbox("I understand how to evaluate end-to-end task completion and proper termination criteria.", value=True)
c4 = st.checkbox("I can analyze agent execution trajectories to detect redundant loops and detour hops.", value=True)
c5 = st.checkbox("I know how to test agent self-correction by injecting synthetic tool crashes.", value=True)
c6 = st.checkbox("I understand how to evaluate intermediate thought coherence in multi-step reasoning agents.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Agent Evaluation, trajectory efficiency auditing, and multi-step reasoning validation in LangChain.")
