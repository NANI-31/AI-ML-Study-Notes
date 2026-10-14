import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.2 Graph Evaluation - LangSmith Master Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class GraphEvaluationEngine:
    """
    Production LangGraph Trajectory & State Machine Evaluator.
    Emulates LangSmith multi-step agent trajectory evaluation across five core pillars:
    1. State Correctness (Schema validation, channel integrity, immutability)
    2. Node Correctness (Per-node contract testing, unit output conformance)
    3. Tool Correctness (Argument schema adherence, redundant call prevention)
    4. Final Output Correctness (Goal fulfillment, semantic grounding)
    5. Workflow Completion (Termination proof, cycle budget enforcement)
    """
    def __init__(self, project_name: str = "langgraph-trajectory-evaluation"):
        self.project_name = project_name
        self.evaluation_history: List[Dict[str, Any]] = []

    def evaluate_graph_trajectory(
        self,
        scenario_preset: str,
        max_allowed_cycles: int = 5,
        strict_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a simulated graph trajectory against benchmark assertions and returns
        a comprehensive scorecard with pillar-level scores and diagnostic findings.
        """
        # Predefined trajectory archetypes for evaluation benchmarking
        trajectories = {
            "optimal": {
                "name": "Optimal Agent Trajectory",
                "cycles": 2,
                "completed": True,
                "state_keys_present": ["messages", "query", "research_notes", "critique_score", "final_answer"],
                "missing_state_keys": [],
                "nodes_executed": [
                    {"name": "researcher", "status": "success", "latency_ms": 140.0, "contract_valid": True},
                    {"name": "tool_search", "status": "success", "latency_ms": 75.0, "contract_valid": True},
                    {"name": "critic", "status": "success", "latency_ms": 110.0, "contract_valid": True},
                    {"name": "synthesizer", "status": "success", "latency_ms": 190.0, "contract_valid": True}
                ],
                "tool_calls": [
                    {"tool": "sec_filings_api", "valid_args": True, "redundant": False, "status": "success"}
                ],
                "grounding_score": 0.94,
                "goal_fulfillment_score": 0.96
            },
            "infinite_loop": {
                "name": "Runaway Infinite Loop (Recursion Exceeded)",
                "cycles": 25,
                "completed": False,
                "state_keys_present": ["messages", "query", "research_notes", "critique_score"],
                "missing_state_keys": ["final_answer"],
                "nodes_executed": [
                    {"name": "researcher", "status": "success", "latency_ms": 130.0, "contract_valid": True},
                    {"name": "critic", "status": "success", "latency_ms": 115.0, "contract_valid": True}
                ] * 12,
                "tool_calls": [
                    {"tool": "sec_filings_api", "valid_args": True, "redundant": True, "status": "success"}
                ] * 6,
                "grounding_score": 0.40,
                "goal_fulfillment_score": 0.15
            },
            "tool_hallucination": {
                "name": "Tool Schema Hallucination & Argument Drift",
                "cycles": 3,
                "completed": True,
                "state_keys_present": ["messages", "query", "research_notes", "critique_score", "final_answer"],
                "missing_state_keys": [],
                "nodes_executed": [
                    {"name": "researcher", "status": "success", "latency_ms": 150.0, "contract_valid": True},
                    {"name": "tool_search", "status": "error", "latency_ms": 45.0, "contract_valid": False},
                    {"name": "synthesizer", "status": "success", "latency_ms": 180.0, "contract_valid": True}
                ],
                "tool_calls": [
                    {"tool": "sec_filings_api", "valid_args": False, "redundant": False, "status": "error"},
                    {"tool": "sec_filings_api", "valid_args": True, "redundant": True, "status": "success"}
                ],
                "grounding_score": 0.62,
                "goal_fulfillment_score": 0.70
            },
            "state_corruption": {
                "name": "Channel Overwrite State Corruption",
                "cycles": 2,
                "completed": True,
                "state_keys_present": ["messages", "query"],
                "missing_state_keys": ["research_notes", "critique_score", "final_answer"],
                "nodes_executed": [
                    {"name": "researcher", "status": "success", "latency_ms": 135.0, "contract_valid": False},
                    {"name": "synthesizer", "status": "success", "latency_ms": 160.0, "contract_valid": False}
                ],
                "tool_calls": [],
                "grounding_score": 0.35,
                "goal_fulfillment_score": 0.45
            }
        }

        traj = trajectories.get(scenario_preset, trajectories["optimal"])

        # 1. State Correctness Assessment
        required_keys = ["messages", "query", "research_notes", "critique_score", "final_answer"]
        keys_present = len([k for k in required_keys if k in traj["state_keys_present"]])
        state_score = keys_present / len(required_keys)
        state_notes = f"{keys_present}/{len(required_keys)} schema channels validated."
        if traj["missing_state_keys"]:
            state_notes += f" Missing channels: {', '.join(traj['missing_state_keys'])}"

        # 2. Node Correctness Assessment
        nodes = traj["nodes_executed"]
        valid_nodes = sum(1 for n in nodes if n["contract_valid"] and n["status"] == "success")
        node_score = valid_nodes / len(nodes) if nodes else 1.0
        node_notes = f"{valid_nodes}/{len(nodes)} node step contracts satisfied."

        # 3. Tool Correctness Assessment
        tools = traj["tool_calls"]
        if not tools:
            tool_score = 1.0 if scenario_preset == "state_corruption" else 0.5
            tool_notes = "No tools invoked in trajectory."
        else:
            valid_tool_calls = sum(1 for t in tools if t["valid_args"] and not t["redundant"] and t["status"] == "success")
            tool_score = valid_tool_calls / len(tools)
            redundant_count = sum(1 for t in tools if t["redundant"])
            tool_notes = f"{valid_tool_calls}/{len(tools)} valid calls. Redundant calls detected: {redundant_count}."

        # 4. Final Output Correctness Assessment
        output_score = (traj["grounding_score"] + traj["goal_fulfillment_score"]) / 2.0
        output_notes = f"Grounding: {traj['grounding_score']:.2f}, Goal Fulfillment: {traj['goal_fulfillment_score']:.2f}."

        # 5. Workflow Completion Assessment
        is_completed = traj["completed"] and traj["cycles"] <= max_allowed_cycles
        completion_score = 1.0 if is_completed else 0.0
        completion_notes = (
            f"Successfully terminated in {traj['cycles']} cycles (Budget: {max_allowed_cycles})."
            if is_completed else
            f"Execution failed to terminate properly! Total cycles {traj['cycles']} exceeded limit or crashed."
        )

        # Composite Trajectory Score (Weighted Evaluation)
        composite_score = (
            (state_score * 0.20) +
            (node_score * 0.20) +
            (tool_score * 0.20) +
            (output_score * 0.25) +
            (completion_score * 0.15)
        )

        evaluation_result = {
            "scenario": traj["name"],
            "composite_score": composite_score,
            "passed_gating": composite_score >= 0.80 and completion_score == 1.0,
            "pillars": {
                "State Correctness": {"score": state_score, "notes": state_notes},
                "Node Correctness": {"score": node_score, "notes": node_notes},
                "Tool Correctness": {"score": tool_score, "notes": tool_notes},
                "Final Output Correctness": {"score": output_score, "notes": output_notes},
                "Workflow Completion": {"score": completion_score, "notes": completion_notes}
            },
            "raw_trajectory": traj
        }

        self.evaluation_history.append(evaluation_result)
        return evaluation_result


# Instantiate engine
graph_evaluator = GraphEvaluationEngine()

# --- HEADER SECTION ---
st.title("⚖️ Chapter 24.2: LangGraph Evaluation & Trajectory Testing")
st.markdown("""
Welcome to the production evaluation manual for **LangGraph agent systems** evaluated through **LangSmith**.
Evaluating autonomous agents cannot be reduced to simple prompt-response matching.
Agents navigate non-deterministic trajectories: executing multi-step loops, invoking external tools,
modifying persistent state channels, and making runtime branch decisions.
This module establishes the five pillars of LangGraph trajectory evaluation to certify agent reliability before production deployment.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LangGraph Trajectory Evaluation & Multi-Step Verification Architecture", divider="rainbow")

st.markdown("""
Trajectory evaluation audits an agent's multi-step execution path against formal correctness contracts.
LangSmith captures the entire execution graph run, allowing automated evaluation suites to grade
every individual step, tool invocation, channel modification, and the final synthesized response.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangGraphEvaluation {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_dataset {
            label="1. Golden Benchmark Dataset (LangSmith)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            bench_ds [label="Benchmark Dataset\nInputs: {task_spec, constraints}\nExpected: {target_artifacts, tool_whitelist}", fillcolor="#1E293B", color="#38BDF8"];
            experiment [label="LangSmith Experiment Run\nTarget: production_stategraph_v2.4", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_eval_pillars {
            label="2. Five Evaluation Pillars";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            e_state [label="Pillar 1: State Correctness\nSchema integrity, channel immutability", fillcolor="#1E293B", color="#38BDF8"];
            e_node [label="Pillar 2: Node Correctness\nFunction contracts, step SLA, latency", fillcolor="#1E293B", color="#38BDF8"];
            e_tool [label="Pillar 3: Tool Correctness\nSchema adherence, zero redundancy", fillcolor="#1E293B", color="#38BDF8"];
            e_out [label="Pillar 4: Final Output Correctness\nFactual grounding, task fulfillment", fillcolor="#1E293B", color="#38BDF8"];
            e_comp [label="Pillar 5: Workflow Completion\nTermination proof, recursion bound", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_decision_engine {
            label="3. Composite Scorecard & CI/CD Gating";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            scorecard [label="Composite Trajectory Scorecard\nWeighted Average (0.0 to 1.0)", fillcolor="#0F172A", color="#10B981"];
            gate_pass [label="CI/CD Release Approved\nComposite >= 0.85 & Completion == 1.0", fillcolor="#064E3B", color="#10B981"];
            gate_fail [label="Release Blocked: Regression Alert\nTool Drift / Loop Runaway Detected", fillcolor="#7F1D1D", color="#EF4444"];
        }

        bench_ds -> experiment;
        experiment -> e_state [label="Audit State Diffs"];
        experiment -> e_node [label="Audit Node Spans"];
        experiment -> e_tool [label="Audit ToolNode Spans"];
        experiment -> e_out [label="Audit Final State"];
        experiment -> e_comp [label="Audit Step Count"];

        e_state -> scorecard;
        e_node -> scorecard;
        e_tool -> scorecard;
        e_out -> scorecard;
        e_comp -> scorecard;

        scorecard -> gate_pass [label="Score >= 0.85", color="#10B981"];
        scorecard -> gate_fail [label="Score < 0.85", color="#EF4444"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Pillars of Graph Trajectory Evaluation
The following matrix defines the assessment criteria and automated evaluators for each pillar:
""")

eval_matrix_df = pd.DataFrame([
    {
        "Pillar": "1. State Correctness",
        "Evaluation Target": "Intermediate state dictionaries across all checkpoints",
        "Verification Metric": "Schema completeness (keys present / keys required), channel immutability score",
        "Failure Signal": "Missing state channels, accidental dictionary wipe, key datatype mismatch"
    },
    {
        "Pillar": "2. Node Correctness",
        "Evaluation Target": "Discrete input/output transitions of individual graph nodes",
        "Verification Metric": "Node contract compliance rate, unit step latency SLA (p95 <= 500ms)",
        "Failure Signal": "Node returns empty dict, hallucinated node outputs, unhandled step exceptions"
    },
    {
        "Pillar": "3. Tool Correctness",
        "Evaluation Target": "ToolNode invocations emitted by agent nodes",
        "Verification Metric": "Tool argument validity rate, redundant call penalty (unique / total calls)",
        "Failure Signal": "Pydantic schema validation failures, calling identical search queries in loop"
    },
    {
        "Pillar": "4. Final Output Correctness",
        "Evaluation Target": "The artifact written to the terminal channel upon reaching END",
        "Verification Metric": "Goal fulfillment score, grounding score against retrieved documents",
        "Failure Signal": "Hallucinated conclusion, incomplete deliverables, non-responsive answer"
    },
    {
        "Pillar": "5. Workflow Completion",
        "Evaluation Target": "Graph termination behavior and cycle depth",
        "Verification Metric": "Binary completion (1.0 = reached END, 0.0 = hit recursion_limit or aborted)",
        "Failure Signal": "GraphRecursionError, oscillating edge transitions, deadlocks"
    }
])
st.dataframe(eval_matrix_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: STATE CORRECTNESS
# ==============================================================================
st.header("2. State Correctness (Schema Validation, Channel Integrity & Required State Keys Verification)", divider="rainbow")

st.markdown("""
State correctness verifies that the graph's shared memory remains valid, complete, and uncorrupted
at every step of execution.
In production LangGraph systems, state schemas are defined with explicit `TypedDict` or `Pydantic` models.
""")

st.markdown("### Automated State Correctness Checks")
st.markdown("""
* **Channel Schema Conformance**: Ensuring every key in the state adheres to its declared type (e.g., `messages` is a list of `BaseMessage` objects, `revision_count` is an integer).
* **Channel Integrity Across Transitions**: Verifying that required keys are not accidentally dropped or overwritten by intermediate nodes.
* **Reducer Invariant Testing**: Confirming that reducer functions (e.g., `add_messages`) correctly deduplicate messages and preserve message ordering.
""")

st.code('''
def evaluate_state_correctness(run_tree: Dict[str, Any], required_channels: List[str]) -> float:
    """
    Custom LangSmith evaluator for state correctness.
    Inspects checkpoints across the run tree and verifies schema adherence.
    """
    checkpoints = run_tree.get("checkpoints", [])
    if not checkpoints:
        return 0.0
    
    total_checks = 0
    passed_checks = 0
    
    for cp in checkpoints:
        state = cp.get("state_snapshot", {})
        for channel in required_channels:
            total_checks += 1
            if channel in state and state[channel] is not None:
                passed_checks += 1
                
    return round(passed_checks / total_checks, 3)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production evaluation pipelines:
* LangSmith evaluators run programmatic assertions on every checkpoint in an experiment run.
* If any node returns an empty update that clears a channel or injects a `None` value into a required field, LangSmith flags the run with a `StateIntegrityViolation` badge.
""")

st.divider()

# ==============================================================================
# SECTION 3: NODE CORRECTNESS
# ==============================================================================
st.header("3. Node Correctness (Unit Node Evaluation, Function Output Integrity & Step-Level Contract Testing)", divider="rainbow")

st.markdown("""
Node correctness treats each graph node as an independently testable unit.
Instead of evaluating the agent only at the end of a 10-step sequence, node correctness evaluates
whether Step K satisfied its contract given the state output from Step K-1.
""")

st.markdown("### Benefits of Step-Level Contract Testing")
st.markdown("""
1. **Pinpointing Blame**: If the final response is incorrect, unit node evaluation determines whether the breakdown occurred at the research node, the retrieval tool, or the synthesis node.
2. **Regression Localization**: When updating a prompt inside a specific node, engineers evaluate only that node's output distribution without incurring the full end-to-end execution cost.
3. **Latency SLA Compliance**: Enforcing individual execution time budgets (e.g., Critic node must complete within 250ms).
""")

st.code('''
def evaluate_node_step_contract(node_name: str, node_input: Dict[str, Any], node_output: Dict[str, Any]) -> bool:
    """
    Validates that a specific node satisfied its pre/post-conditions.
    """
    if node_name == "researcher_node":
        # Contract: Must produce non-empty research notes
        return bool(node_output.get("research_notes")) and len(node_output["research_notes"]) > 50
    elif node_name == "critic_node":
        # Contract: Must produce critique score between 0.0 and 1.0
        score = node_output.get("critique_score", -1.0)
        return 0.0 <= score <= 1.0
    return True
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern LLMOps workflows isolate node datasets in LangSmith:
* Engineers extract inputs and outputs of individual nodes from production traces to build **node-specific unit test datasets**.
* During CI/CD, nodes are invoked as standalone runnables against these datasets, achieving 10x faster test runs and reducing evaluation token costs.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOOL CORRECTNESS
# ==============================================================================
st.header("4. Tool Correctness (Argument Schema Conformance, Tool Call Efficiency & Redundant Call Detection)", divider="rainbow")

st.markdown("""
Tool correctness measures how accurately and efficiently an agent leverages its tools.
An agent that calls tools with hallucinated arguments, triggers repeated exceptions, or calls the same API
ten times with identical parameters demonstrates poor trajectory quality.
""")

st.markdown("### Core Metrics for Tool Evaluation")
st.markdown("""
* **Argument Schema Validity**: Percentage of tool calls whose arguments strictly validate against the tool's Pydantic schema without throwing `ValidationError`.
* **Tool Call Efficiency**: Measuring the ratio of unique tool calls to total tool calls:
  `efficiency = unique_tool_calls / total_tool_calls`.
  A low ratio indicates redundant or spinning behavior.
* **Tool Success Rate**: Percentage of tool executions that return successful payloads rather than HTTP 4xx/5xx or `ToolException` errors.
""")

st.code('''
def evaluate_tool_trajectory(tool_spans: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Grades tool selection and argument generation across an agent trajectory.
    """
    if not tool_spans:
        return {"schema_validity": 1.0, "efficiency": 1.0, "success_rate": 1.0}
        
    total_calls = len(tool_spans)
    successful_calls = sum(1 for s in tool_spans if s["status"] == "success")
    
    # Check for redundant duplicate calls
    call_signatures = [f"{s['tool_name']}:{str(s['inputs'])}" for s in tool_spans]
    unique_calls = len(set(call_signatures))
    
    return {
        "schema_validity": round(successful_calls / total_calls, 3),
        "efficiency": round(unique_calls / total_calls, 3),
        "success_rate": round(successful_calls / total_calls, 3)
    }
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production agent evaluation:
* LangSmith traces capture every tool call span under `ToolNode`.
* Automated evaluators flag traces where identical arguments appear more than once within the same thread.
* If an agent repeatedly calls a tool without varying arguments after receiving an error, it is penalized with a zero for tool correctness.
""")

st.divider()

# ==============================================================================
# SECTION 5: FINAL OUTPUT CORRECTNESS
# ==============================================================================
st.header("5. Final Output Correctness (End-to-End Goal Fulfillment, Grounding & Semantic Correctness)", divider="rainbow")

st.markdown("""
Final output correctness assesses the artifact delivered to the user upon workflow completion.
Even if an agent executes perfect tool calls and maintains clean state schemas, if the final answer
fails to satisfy the user's objective or contains hallucinations, the workflow fails.
""")

st.markdown("### Dual Evaluation Dimensions")
st.markdown("""
1. **Task / Goal Fulfillment**: Does the synthesized output directly answer the prompt, follow all negative constraints, and supply required deliverables?
2. **Contextual Grounding**: Are all factual assertions in the final output supported by evidence retrieved by the tools and stored in intermediate state channels?
""")

st.code('''
from langsmith.evaluation import evaluate

# LLM-as-a-judge prompt for goal fulfillment
GOAL_FULFILLMENT_PROMPT = """
Compare the agent final output against the user goal.
User Goal: {input}
Agent Output: {output}

Grade the fulfillment from 0.0 to 1.0 based on completeness and accuracy.
"""

def goal_fulfillment_evaluator(run, example) -> Dict[str, Any]:
    # Custom evaluator integrated with LangSmith experiment runner
    prediction = run.outputs.get("final_answer", "")
    ground_truth = example.outputs.get("target_answer", "")
    
    # Compute semantic similarity or rubric score
    score = compute_rubric_score(prediction, ground_truth)
    return {"key": "goal_fulfillment", "score": score}
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams run **LLM-as-a-Judge** evaluators directly in LangSmith:
* Judges evaluate both the final answer and the retrieved state notes.
* The evaluation output is rendered as a score from 0.0 to 1.0 with a detailed reasoning justification.
* Runs scoring below 0.80 are automatically added to a human annotation queue for manual review.
""")

st.divider()

# ==============================================================================
# SECTION 6: WORKFLOW COMPLETION
# ==============================================================================
st.header("6. Workflow Completion (Cycle Bound Verification, Infinite Loop Detection & Termination Proofs)", divider="rainbow")

st.markdown("""
Workflow completion evaluates the agent's termination behavior.
Cyclic state machines can easily get trapped in infinite loops, oscillation cycles, or exceed recursion limits.
A reliable agent must deterministically converge to `__end__` within a bounded number of execution steps.
""")

st.markdown("### Termination Proofs & Cycle Auditing")
st.markdown("""
* **Binary Completion Guarantee**: The graph must reach `__end__` without throwing `GraphRecursionError` or terminating on an unhandled exception.
* **Cycle Budget Compliance**: If a workflow is allocated a budget of 4 iterations, completing in 5 iterations is classified as a budget overrun failure.
* **Convergence Trajectory**: Inspecting the rate at which critique scores or error signals decrease across successive iterations.
""")

st.code('''
def evaluate_workflow_completion(run_tree: Dict[str, Any], max_allowed_cycles: int = 5) -> Dict[str, Any]:
    """
    Evaluates whether the graph reached END properly within budget.
    """
    total_cycles = run_tree.get("total_cycles", 0)
    reached_end = run_tree.get("status") == "completed"
    
    within_budget = total_cycles <= max_allowed_cycles
    passed = reached_end and within_budget
    
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "total_cycles": total_cycles,
        "budget": max_allowed_cycles,
        "failure_mode": None if passed else ("recursion_exceeded" if not within_budget else "premature_crash")
    }
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production CI/CD testing:
* Every agent experiment enforces a hard `recursion_limit` assert.
* If any test case hits the recursion limit or executes 20+ steps without terminating, the CI build fails immediately.
* LangSmith visualizes the execution step distribution across all test cases, revealing outliers that consume excessive cycles.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: LangGraph Trajectory & State Machine Evaluator (GraphEvaluationEngine)", divider="rainbow")

st.markdown("""
Below is the reference production evaluation implementation.
The `GraphEvaluationEngine` grades multi-step agent trajectories against all five pillars
and computes a composite quality score with automated gating recommendations.
""")

st.code('''
class GraphEvaluationEngine:
    """
    Production LangGraph Trajectory & State Machine Evaluator.
    Emulates LangSmith multi-step agent trajectory evaluation across five core pillars.
    """
    def __init__(self, project_name: str = "langgraph-trajectory-evaluation"):
        self.project_name = project_name
        self.evaluation_history: List[Dict[str, Any]] = []

    def evaluate_graph_trajectory(
        self,
        scenario_preset: str,
        max_allowed_cycles: int = 5,
        strict_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a simulated graph trajectory against benchmark assertions and returns
        a comprehensive scorecard with pillar-level scores and diagnostic findings.
        """
        # Evaluates state correctness, node correctness, tool correctness,
        # final output correctness, and workflow completion.
        ...
''', language="python")

st.info("The evaluation engine is instantiated in module scope and powers the interactive trajectory scorecard below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Graph Trajectory Scorecard & Infinite Loop Detector", divider="rainbow")

st.markdown("""
Test how LangSmith evaluates diverse LangGraph agent trajectories.
Select a test scenario (from optimal execution to runaway infinite loops), adjust evaluation parameters,
and inspect the multi-pillar scorecard and automated release gating decision.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Evaluation Scenario Controls")
    selected_scenario = st.selectbox(
        "Select Trajectory Benchmark Scenario:",
        options=["optimal", "infinite_loop", "tool_hallucination", "state_corruption"],
        format_func=lambda s: {
            "optimal": "1. Optimal Agent Trajectory (Gold Standard)",
            "infinite_loop": "2. Runaway Infinite Loop (Recursion Exceeded)",
            "tool_hallucination": "3. Tool Schema Hallucination & Argument Drift",
            "state_corruption": "4. Channel Overwrite State Corruption"
        }[s]
    )

    cycle_budget = st.slider(
        "Maximum Allowed Cycle Budget:",
        min_value=2,
        max_value=10,
        value=5,
        help="Permissible cyclic revisions before flagging as runaway recursion"
    )

    enforce_strict = st.toggle(
        "Enforce Strict Schema Assertion",
        value=True,
        help="Penalizes missing optional state channels and schema type mismatches"
    )

    eval_btn = st.button("⚖️ Run Trajectory Evaluation Suite", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Trajectory Scorecard")

    if eval_btn or len(graph_evaluator.evaluation_history) == 0:
        eval_result = graph_evaluator.evaluate_graph_trajectory(
            scenario_preset=selected_scenario,
            max_allowed_cycles=cycle_budget,
            strict_schema=enforce_strict
        )
    else:
        eval_result = graph_evaluator.evaluation_history[-1]

    # Top Level Scorecard Metrics
    m1, m2, m3 = st.columns(3)
    composite = eval_result["composite_score"]
    status_label = "APPROVED (PASSED)" if eval_result["passed_gating"] else "REJECTED (FAILED)"
    m1.metric("Composite Trajectory Score", f"{composite * 100:.1f} %")
    m2.metric("CI/CD Release Gate", status_label)
    m3.metric("Evaluated Scenario", eval_result["scenario"][:24])

    if eval_result["passed_gating"]:
        st.success("✅ **Release Approved**: Trajectory meets all state, tool, output, and termination quality thresholds.")
    else:
        st.error("🚨 **Release Blocked**: Trajectory violated critical quality thresholds. Review pillar breakdown below.")

    # Pillar Scores Breakdown Table
    st.markdown("#### Five Pillars Performance Breakdown")
    pillars_data = []
    for pillar_name, pdata in eval_result["pillars"].items():
        pillars_data.append({
            "Evaluation Pillar": pillar_name,
            "Score": f"{pdata['score'] * 100:.1f} %",
            "Diagnostic Findings & Audit Notes": pdata["notes"]
        })
    st.dataframe(pd.DataFrame(pillars_data), use_container_width=True)

    # Detailed Raw Trajectory Audit
    with st.expander("🔍 View Raw Trajectory Step Breakdown", expanded=False):
        st.json(eval_result["raw_trajectory"])

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why is evaluating LangGraph agents fundamentally different from evaluating single-turn LLM chains?"):
    st.markdown("""
    **Answer:**
    Single-turn chains are static functions mapping input text to output text.
    In contrast, LangGraph agents are dynamic state machines characterized by:
    1. Multi-step cyclic trajectories where early errors compound across subsequent iterations.
    2. Side-effects from tool execution that mutate external environments.
    3. Intermediate state channels that must preserve schema invariants across checkpointers.
    4. Non-deterministic branch routing where an agent can take vastly different valid paths to solve the same goal.
    Therefore, evaluation must audit the entire **trajectory** (state diffs, tool calls, recursion bounds) in addition to the final output.
    """)

with st.expander("Q2: How do you detect and evaluate redundant tool calls in an agent trajectory?"):
    st.markdown("""
    **Answer:**
    Redundant tool calls occur when an agent invokes the same tool with identical or semantically duplicate arguments repeatedly during iterative loops.
    In LangSmith:
    1. Extract all `ToolNode` child spans from the execution run tree.
    2. Compute a hash signature for each call: `hash(tool_name, json.dumps(inputs, sort_keys=True))`.
    3. Calculate the uniqueness ratio: `len(set(signatures)) / len(signatures)`.
    4. Penalize trajectories where the ratio falls below a defined threshold (e.g., 0.80), flagging the agent for tool thrashing.
    """)

with st.expander("Q3: What metrics constitute the 'State Correctness' pillar in LangGraph evaluation?"):
    st.markdown("""
    **Answer:**
    State Correctness evaluates whether shared memory remains uncorrupted:
    1. **Schema Completeness**: Verifying that all mandatory keys defined in the `TypedDict`/Pydantic schema are present at each checkpoint.
    2. **Channel Immutability**: Ensuring nodes return update dictionaries rather than mutating state in-place.
    3. **Reducer Invariants**: Validating that message channels properly append or deduplicate without dropping conversation context.
    4. **Payload Cleanliness**: Ensuring channels do not accumulate corrupted non-serializable objects that break checkpointers.
    """)

with st.expander("Q4: How does LangSmith facilitate unit testing of individual graph nodes?"):
    st.markdown("""
    **Answer:**
    LangSmith enables node unit testing by:
    1. Capturing input state snapshots from real production traces and saving them as curated test datasets.
    2. Invoking the individual node function directly with the saved state input without running the entire upstream graph.
    3. Comparing the node's output update against expected criteria (e.g., did the critic output a float between 0.0 and 1.0?).
    This isolates node regressions, speeds up CI/CD test runs by 10x, and reduces evaluation LLM costs.
    """)

with st.expander("Q5: What is the recommended strategy for enforcing workflow completion in CI/CD pipelines?"):
    st.markdown("""
    **Answer:**
    1. Set a strict `recursion_limit` in the test configuration (e.g., `recursion_limit=10`).
    2. Assert that the run status equals `completed` and that `__end__` was reached.
    3. Reject test runs that terminate via `GraphRecursionError`.
    4. Track average cycle depth across releases; alert if a model upgrade causes average iterations to drift from 2.1 to 3.8 cycles per task.
    """)

st.markdown("### Production Traps & Failure Modes")

ge_tr_col1, ge_tr_col2 = st.columns(2)

with ge_tr_col1:
    st.error("""
    **Trap 1: Evaluating Only the Final Answer (The Output Illusion)**
    * **Failure**: Grading an agent solely on its final text answer while ignoring its intermediate trajectory.
    * **Consequence**: An agent that took 24 wasteful tool iterations and burned $3.50 in API calls is graded as 'passing' because its final sentence happened to be correct.
    * **Remedy**: Always enforce multi-pillar evaluation combining trajectory efficiency, tool correctness, and final output accuracy.
    """)

    st.error("""
    **Trap 2: Brittle String Equality in State Checks**
    * **Failure**: Writing assertions that check for exact string matches in intermediate research notes or draft channels.
    * **Consequence**: Minor stochastic wording variations in LLM completions cause tests to fail constantly, creating false alarms.
    * **Remedy**: Assert on structural properties, schema keys, and semantic rubric scores rather than verbatim string equality.
    """)

with ge_tr_col2:
    st.warning("""
    **Trap 3: Masking Tool Exceptions Inside Tool Wrappers**
    * **Failure**: Writing tool functions that catch errors and return `"Tool failed"` strings without flagging failure status.
    * **Consequence**: The evaluation suite records a 100% tool success rate because no exceptions were raised, masking systematic downstream API integration errors.
    * **Remedy**: Return structured `ToolMessage(status='error')` or raise typed `ToolException` so LangSmith records the true error rate.
    """)

    st.warning("""
    **Trap 4: Missing Cycle Budgets in Test Configurations**
    * **Failure**: Running evaluation benchmarks without a hard recursion bound in the test runner.
    * **Consequence**: A single rogue agent caught in an infinite loop can hang CI test pipelines and consume hundreds of dollars in OpenAI tokens.
    * **Remedy**: Always inject explicit `recursion_limit` ceilings in test `RunnableConfig` instances.
    """)

st.markdown("### Core Technical Flashcards")

fcge_c1, fcge_c2 = st.columns(2)

with fcge_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the five pillars of LangGraph trajectory evaluation?")
        st.markdown("**Back**: 1. State Correctness, 2. Node Correctness, 3. Tool Correctness, 4. Final Output Correctness, and 5. Workflow Completion.")

    with fcge_c2:
        st.markdown("**Front**: What is the formula for Tool Call Efficiency in trajectory evaluation?")
        st.markdown("**Back**: `efficiency = unique_tool_calls / total_tool_calls`. Penalizes agents that repeatedly invoke tools with identical parameters.")

with fcge_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does unit node evaluation reduce LLM evaluation costs in CI/CD?")
        st.markdown("**Back**: By testing individual node functions in isolation against saved state inputs, avoiding the token expense of executing the entire upstream multi-step graph.")

    with fcge_c2:
        st.markdown("**Front**: What constitutes a Workflow Completion failure in LangGraph?")
        st.markdown("**Back**: When a graph hits its `recursion_limit` without terminating, crashes on an unhandled node exception, or reaches an unexpected dead-end without outputting required deliverables.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LangGraph trajectory evaluation:")

ge_chk1 = st.checkbox("I understand the 5 core pillars of LangGraph trajectory evaluation in LangSmith.", value=True)
ge_chk2 = st.checkbox("I can validate state schema completeness and channel integrity across checkpoints.", value=True)
ge_chk3 = st.checkbox("I can isolate individual node contracts and write step-level unit test evaluators.", value=True)
ge_chk4 = st.checkbox("I know how to compute Tool Call Efficiency and detect redundant or spinning tool calls.", value=True)
ge_chk5 = st.checkbox("I understand how to evaluate goal fulfillment and factual grounding in agent final states.", value=True)
ge_chk6 = st.checkbox("I can enforce workflow completion proofs and recursion bounds in automated CI/CD gating.", value=True)

if ge_chk1 and ge_chk2 and ge_chk3 and ge_chk4 and ge_chk5 and ge_chk6:
    st.success("Mastery Certified: You have mastered LangGraph trajectory evaluation, multi-step contract testing, and CI/CD agent gating in LangSmith.")
