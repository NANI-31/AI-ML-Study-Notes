import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="38.3 Advanced Coding Problems - LangSmith Curriculum",
    page_icon="🚀",
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
    .badge-adv {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-pareto {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class LangSmithAdvancedEngine:
    """
    Simulates enterprise-grade LangSmith capabilities:
    Automated evaluation pipelines, LLM-as-a-judge rubrics,
    agent trajectory verification, regression delta analysis,
    and Pareto cost-performance frontier modeling.
    """
    def __init__(self):
        self.pareto_models = [
            {"model": "gpt-4o-mini", "accuracy": 0.82, "cost_per_1k": 0.0004, "latency_ms": 320, "pareto_optimal": True},
            {"model": "gpt-4o", "accuracy": 0.94, "cost_per_1k": 0.0042, "latency_ms": 940, "pareto_optimal": True},
            {"model": "claude-3-5-sonnet", "accuracy": 0.96, "cost_per_1k": 0.0055, "latency_ms": 1180, "pareto_optimal": True},
            {"model": "legacy-gpt-3.5-turbo", "accuracy": 0.71, "cost_per_1k": 0.0015, "latency_ms": 680, "pareto_optimal": False},
            {"model": "open-mixtral-8x7b", "accuracy": 0.81, "cost_per_1k": 0.0007, "latency_ms": 410, "pareto_optimal": False}
        ]

        self.agent_trajectories = [
            {
                "id": "traj-success-01",
                "goal": "Calculate Q3 tax liability for Delaware enterprise",
                "steps": [
                    {"step": 1, "tool": "search_tax_code", "status": "OK", "latency_ms": 140},
                    {"step": 2, "tool": "sql_query_delaware_revenue", "status": "OK", "latency_ms": 280},
                    {"step": 3, "tool": "calculate_liability", "status": "OK", "latency_ms": 45}
                ],
                "valid_path": True,
                "hallucinated_tools": 0,
                "efficiency_score": 1.00
            },
            {
                "id": "traj-fail-02",
                "goal": "Retrieve invoice and generate PDF receipt",
                "steps": [
                    {"step": 1, "tool": "search_invoice", "status": "OK", "latency_ms": 110},
                    {"step": 2, "tool": "send_email_unauthorized", "status": "ERROR_SCHEMA", "latency_ms": 90},
                    {"step": 3, "tool": "search_invoice", "status": "REDUNDANT", "latency_ms": 115},
                    {"step": 4, "tool": "generate_pdf_receipt", "status": "OK", "latency_ms": 520}
                ],
                "valid_path": False,
                "hallucinated_tools": 1,
                "efficiency_score": 0.50
            }
        ]

    def run_regression_gate(self, baseline_acc: float, candidate_acc: float, delta_thresh: float) -> Dict[str, Any]:
        delta = round(candidate_acc - baseline_acc, 4)
        passed = delta >= -delta_thresh
        return {
            "baseline_accuracy": baseline_acc,
            "candidate_accuracy": candidate_acc,
            "delta": delta,
            "threshold": -delta_thresh,
            "verdict": "APPROVE_MERGE" if passed else "BLOCK_MERGE",
            "passed": passed
        }

    def simulate_llm_judge_score(self, reasoning_depth: str, answer_fidelity: str) -> Dict[str, Any]:
        scores = {
            "Shallow": 0.60,
            "Moderate": 0.82,
            "Rigorous Chain-of-Thought": 0.96
        }
        fidelity_mult = 1.0 if answer_fidelity == "Strict Grounding" else 0.75 if answer_fidelity == "Minor Extrapolation" else 0.35
        raw_score = scores.get(reasoning_depth, 0.70) * fidelity_mult
        return {
            "score": round(raw_score, 2),
            "verdict": "EXCELLENT" if raw_score >= 0.90 else "ACCEPTABLE" if raw_score >= 0.75 else "REJECTED",
            "rationale": f"Evaluated under {reasoning_depth} analysis with {answer_fidelity} criteria."
        }

# Instantiate engine
advanced_engine = LangSmithAdvancedEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Continuous Evaluation, Trajectory Analysis, & Cost-Quality Pareto Frontiers", divider="rainbow")

st.markdown("""
At the enterprise tier, LangSmith operates as the **central nervous system for generative AI governance**. 
This blueprint illustrates the continuous feedback lifecycle: production runs are filtered and sampled, 
multi-agent trajectories are audited for structural invariants, LLM judges grade responses with structured rubrics, 
and Pareto analysis models optimize the exact cost-quality-latency trade-offs across model portfolios.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Advanced_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_prod_sampling {
            label = "Production Stream & Continuous Sampling";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            prod_traffic [label="Production Traffic Stream\\n(Thousands of Live Queries/Day)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            sampling_filter [label="Automated Sampling Rule\\n(10% Random + 100% Thumbs-Down)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            curated_queue [label="Continuous Evaluation Queue\\n(Auto-fed into Golden Datasets)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_eval_engine {
            label = "Advanced Multi-Dimensional Evaluation Engine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            judge_evaluator [label="Structured LLM-as-a-Judge\\n(Pydantic CoT Rubrics, Pairwise Arena)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            rag_deep_eval [label="Multi-Metric RAG Audit\\n(Faithfulness, Context Precision/Recall)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            trajectory_audit [label="Agent Trajectory Validator\\n(Tool Call Sequences, State Invariants)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
        }

        subgraph cluster_decision {
            label = "Continuous Governance & Cost-Performance Pareto Optimization";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            ci_gate [label="CI/CD Regression Blocker\\n(Delta Tolerance Gates in PRs)", fillcolor="#7F1D1D", fontcolor="#F87171", color="#EF4444"];
            pareto_frontier [label="Pareto Frontier Analysis\\n(Accuracy vs Cost vs Latency Optimization)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            model_router [label="Dynamic Semantic Router\\n(Routes Easy Queries to Cheap Models)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        prod_traffic -> sampling_filter;
        sampling_filter -> curated_queue;
        curated_queue -> judge_evaluator;
        curated_queue -> rag_deep_eval;
        curated_queue -> trajectory_audit;

        judge_evaluator -> ci_gate [label="delta scores"];
        rag_deep_eval -> pareto_frontier [label="quality vectors"];
        trajectory_audit -> ci_gate [label="state checks"];

        pareto_frontier -> model_router [label="optimal policy"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Enterprise Paradigm Shift:</strong> Evaluation is no longer a pre-deployment checkpoint; it is a 
    <strong>continuous feedback loop</strong>. Real-world edge cases flow directly into test suites, 
    and model routing algorithms dynamically balance accuracy against token budgets.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: BUILD AN AUTOMATED EVALUATION PIPELINE
# ==============================================================================
st.header("2. Build an Automated Evaluation Pipeline: Scheduled Evaluators and CI/CD Integrations", divider="rainbow")

st.markdown("""
An automated evaluation pipeline runs programmatically without human intervention. 
It connects versioned dataset repositories to CI/CD workflows (such as GitHub Actions or GitLab CI) 
and nightly scheduled cron jobs.
""")

st.code('''from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def automated_ci_eval_pipeline(pipeline_fn, dataset_name: str, evaluators: list):
    """
    Automated evaluation runner executed inside CI/CD workflows.
    """
    print(f"Triggering automated evaluation against: {dataset_name}")
    results = evaluate(
        pipeline_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="ci-automated-gate",
        max_concurrency=4
    )
    
    # Extract aggregate feedback stats
    experiment = client.read_project(project_name=results.experiment_name)
    print(f"Automated evaluation completed. Experiment: {experiment.name}")
    return results
''', language="python")

# ==============================================================================
# SECTION 3: CREATE AN LLM-AS-A-JUDGE EVALUATOR
# ==============================================================================
st.header("3. Create an LLM-as-a-Judge Evaluator: Structured Pydantic Criteria and Chain-of-Thought", divider="rainbow")

st.markdown("""
Ad-hoc text prompts for LLM judges often yield brittle or inconsistent scores. 
Production judges use **structured outputs (Pydantic schemas)** and enforce step-by-step 
Chain-of-Thought (CoT) reasoning before emitting a numerical grade.
""")

st.code('''from typing import Dict, Any
from pydantic import BaseModel, Field
from langsmith.schemas import Run, Example

class JudgeScorecard(BaseModel):
    chain_of_thought: str = Field(description="Step-by-step analysis comparing response to source context.")
    factual_claims_total: int = Field(description="Total factual claims made in the answer.")
    factual_claims_grounded: int = Field(description="Claims explicitly supported by context.")
    score: float = Field(description="Normalized score between 0.0 and 1.0.")
    verdict: str = Field(description="PASS, BORDERLINE, or FAIL.")

def llm_judge_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Evaluates faithfulness using structured output schema.
    """
    query = run.inputs.get("query", "")
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    
    # Prompt template with structured schema enforcement
    judge_prompt = f"""
    Evaluate whether the following answer is faithful to the context.
    Query: {query}
    Context: {context}
    Answer: {answer}
    """
    # structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(JudgeScorecard)
    # result: JudgeScorecard = structured_llm.invoke(judge_prompt)
    
    # Simulating parsed output
    return {
        "key": "llm_judge_faithfulness",
        "score": 0.95,
        "comment": "All 4 factual claims verified in context with zero hallucinations."
    }
''', language="python")

# ==============================================================================
# SECTION 4: BUILD RAG EVALUATION
# ==============================================================================
st.header("4. Build RAG Evaluation: Multi-Metric Assessment Across Precision, Recall, and Faithfulness", divider="rainbow")

st.markdown("""
A comprehensive RAG audit requires a suite of distinct evaluators that isolate 
every failure point across the retrieval and generation phases.
""")

rag_metrics_df = pd.DataFrame([
    {"Metric": "Context Precision", "Sub-System": "Retriever", "Formula": "Relevant Chunks at Top Ranks / Total Retrieved", "Significance": "Ensures relevant context appears at the top to avoid 'lost in the middle' syndrome."},
    {"Metric": "Context Recall", "Sub-System": "Retriever", "Formula": "Retrieved Ground Truth Claims / Total Ground Truth Claims", "Significance": "Detects if retriever omitted necessary evidence required to answer complex multi-hop queries."},
    {"Metric": "Faithfulness", "Sub-System": "Generator", "Formula": "Supported Claims / Total Answer Claims", "Significance": "Guarantees zero model hallucinations."},
    {"Metric": "Answer Relevance", "Sub-System": "End-to-End", "Formula": "Semantic Alignment with User Query", "Significance": "Prevents model from generating truthful but completely irrelevant answers."}
])
st.dataframe(rag_metrics_df, use_container_width=True)

# ==============================================================================
# SECTION 5: BUILD AGENT TRAJECTORY EVALUATION
# ==============================================================================
st.header("5. Build Agent Trajectory Evaluation: Tool Call Sequences and State Invariants", divider="rainbow")

st.markdown("""
Evaluating autonomous agents requires validating the **order of operations**, verifying that 
no unauthorized or hallucinated tools were invoked, and ensuring state invariants remain intact.
""")

st.code('''def evaluate_agent_trajectory(run: Run, example: Example) -> Dict[str, Any]:
    """
    Audits the execution path of a LangGraph agent.
    """
    # Extract child spans representing tool invocations
    tool_spans = [s for s in run.child_runs if s.run_type == "tool"]
    tool_names = [s.name for s in tool_spans]
    
    expected_tools = example.outputs.get("required_tools", [])
    forbidden_tools = ["execute_arbitrary_shell", "delete_database"]
    
    # Check 1: Forbidden tool invocation
    has_forbidden = any(t in forbidden_tools for t in tool_names)
    if has_forbidden:
        return {"key": "trajectory_safety", "score": 0.0, "comment": "Agent invoked forbidden tool!"}
        
    # Check 2: Required tool coverage
    coverage = sum(1 for t in expected_tools if t in tool_names) / max(len(expected_tools), 1)
    
    # Check 3: Efficiency penalty for redundant steps
    redundancy_penalty = 0.1 * max(len(tool_names) - len(expected_tools), 0)
    final_score = max(round(coverage - redundancy_penalty, 2), 0.0)
    
    return {
        "key": "trajectory_efficiency",
        "score": final_score,
        "comment": f"Invoked tools: {tool_names}. Coverage: {coverage}, Penalty: {redundancy_penalty}"
    }
''', language="python")

# ==============================================================================
# SECTION 6: CREATE REGRESSION TESTING
# ==============================================================================
st.header("6. Create Regression Testing: Baseline Comparisons and Automated PR Gates", divider="rainbow")

st.markdown("""
Regression testing compares a new pull request's candidate experiment against an established 
production baseline experiment in LangSmith.
""")

st.code('''from langsmith import Client

client = Client()

def run_regression_quality_gate(candidate_exp_name: str, baseline_exp_name: str, tolerance: float = 0.02):
    """
    Asserts candidate experiment does not regress by more than tolerance (2%).
    """
    cand = client.get_test_run(experiment_name=candidate_exp_name)
    base = client.get_test_run(experiment_name=baseline_exp_name)
    
    cand_score = cand.feedback_stats.get("correctness", {}).get("mean", 0.0)
    base_score = base.feedback_stats.get("correctness", {}).get("mean", 0.0)
    
    delta = cand_score - base_score
    print(f"Baseline Score:  {base_score:.4f}")
    print(f"Candidate Score: {cand_score:.4f}")
    print(f"Delta:           {delta:+.4f} (Allowed tolerance: -{tolerance})")
    
    if delta < -tolerance:
        raise AssertionError(f"QUALITY REGRESSION DETECTED: Score dropped by {abs(delta):.2%}, exceeding tolerance {tolerance:.2%}")
    print("Quality gate passed! Safe to merge.")
''', language="python")

# ==============================================================================
# SECTION 7: MONITOR PRODUCTION QUALITY
# ==============================================================================
st.header("7. Monitor Production Quality: Online Sampling and Automated Feedback Rules", divider="rainbow")

st.markdown("""
In production, running LLM judges on 100% of traffic is cost-prohibitive. 
Production monitoring architectures use **intelligent sampling rules**:
""")

st.markdown("""
1. **Statistical Shadow Sampling**: Route a deterministic 5% to 10% slice of user traffic to an asynchronous evaluation worker.
2. **Negative Signal Capture (100% Audit)**: Any session that triggers user thumbs-down, regeneration, or an HTTP 500 error is immediately flagged for full trace auditing.
3. **Automated Guardrail Evaluators**: Lightweight classifiers (regex, toxicity models, length detectors) run on 100% of outputs at microsecond latency.
""")

# ==============================================================================
# SECTION 8: BUILD COST-PERFORMANCE ANALYSIS
# ==============================================================================
st.header("8. Build Cost-Performance Analysis: Pareto Efficiency Frontiers", divider="rainbow")

st.markdown("""
Engineering leaders must make data-driven decisions: *Is a 2% improvement in accuracy worth a 10x increase in token cost and 3x higher latency?* 
The **Pareto Frontier** identifies models that are mathematically optimal—where no other model achieves higher accuracy at lower cost.
""")

# ==============================================================================
# SECTION 9: BUILD CONTINUOUS EVALUATION
# ==============================================================================
st.header("9. Build Continuous Evaluation: Active Learning Loops and Production Backfills", divider="rainbow")

st.markdown("""
Continuous evaluation turns production into an active learning engine. 
Failed production queries are automatically sanitized, converted into golden dataset format, 
and queued for human annotation.
""")

st.code('''def continuous_dataset_backfill(failed_run_id: str, target_dataset: str):
    """
    Extracts inputs from a failed production trace and backfills into golden benchmark.
    """
    run = client.read_run(run_id=failed_run_id)
    
    # Anonymize PII and sanitize inputs
    clean_inputs = {"query": run.inputs.get("query", "")}
    
    # Ingest into dataset with 'pending_review' tag
    client.create_example(
        inputs=clean_inputs,
        outputs={"expected_answer": "PENDING_HUMAN_ANNOTATION"},
        dataset_name=target_dataset,
        metadata={"source": "production_failure_backfill", "original_run_id": failed_run_id}
    )
    print(f"Run {failed_run_id} backfilled into {target_dataset} for review.")
''', language="python")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Enterprise Continuous Evaluation & Production Monitoring Suite", divider="rainbow")

st.markdown("""
Below is the complete enterprise-grade Python implementation demonstrating continuous evaluation 
sampling, trajectory auditing, and delta threshold assertion.
""")

st.code('''import time
from typing import Dict, Any, List
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

client = Client()

# 1. Multi-Dimensional Enterprise Evaluator
def enterprise_quality_evaluator(run, example) -> List[Dict[str, Any]]:
    output = run.outputs.get("answer", "")
    target = example.outputs.get("expected", "")
    
    # Metric 1: Correctness
    is_correct = 1.0 if target.lower() in output.lower() else 0.0
    
    # Metric 2: Latency Compliance (SLO <= 1500ms)
    duration_s = (run.end_time - run.start_time).total_seconds() if run.end_time else 0.5
    latency_pass = 1.0 if duration_s <= 1.5 else 0.0
    
    return [
        {"key": "correctness", "score": is_correct},
        {"key": "latency_slo_compliance", "score": latency_pass, "comment": f"Duration: {duration_s:.3f}s"}
    ]

# 2. Target Agent Pipeline Under Evaluation
@traceable(run_type="chain", name="enterprise_rag_agent")
def enterprise_rag_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    time.sleep(0.12)
    query = inputs.get("query", "")
    return {
        "answer": f"Enterprise answer addressing: {query}",
        "context": "Enterprise documentation SLA section 5."
    }

# 3. Execution & Regression Check
DATASET = "enterprise-continuous-eval-v1"

if not client.has_dataset(dataset_name=DATASET):
    ds = client.create_dataset(dataset_name=DATASET)
    client.create_examples(
        inputs=[{"query": "What is the uptime SLA?"}],
        outputs=[{"expected": "uptime SLA"}],
        dataset_id=ds.id
    )

results = evaluate(
    enterprise_rag_agent,
    data=DATASET,
    evaluators=[enterprise_quality_evaluator],
    experiment_prefix="enterprise-ci-eval"
)
print(f"Enterprise Evaluation Run Finished: {results.experiment_name}")
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Enterprise Evaluation, Trajectory Audit & Pareto Frontier Simulator", divider="rainbow")

st.markdown("""
Experiment with the interactive tools below to simulate LLM-as-a-judge scoring, 
audit agent tool call sequences, and analyze cost-accuracy Pareto frontiers.
""")

wb_view = st.selectbox(
    "Select Advanced Engineering Tool:",
    ["1. Cost-Performance Pareto Frontier Analyzer", "2. Agent Trajectory Sequence Auditor", "3. CI/CD Regression Gate Simulator"]
)

if wb_view == "1. Cost-Performance Pareto Frontier Analyzer":
    st.subheader("Model Pareto Efficiency Matrix")
    p_df = pd.DataFrame(advanced_engine.pareto_models)
    st.dataframe(p_df, use_container_width=True)

    st.markdown("#### Strategic Takeaways:")
    st.info("""
    - **gpt-4o-mini**: Pareto-optimal for low-cost, high-volume classification (82% accuracy at $0.0004/1k).
    - **gpt-4o**: Pareto-optimal champion for complex extraction and general reasoning (94% accuracy).
    - **legacy-gpt-3.5-turbo**: Pareto-dominated (inferior accuracy and higher cost than gpt-4o-mini). Deprecate immediately.
    """)

elif wb_view == "2. Agent Trajectory Sequence Auditor":
    st.subheader("Agent Tool Sequence Audit")
    traj_idx = st.selectbox("Select Trajectory to Audit:", [0, 1], format_func=lambda x: f"{advanced_engine.agent_trajectories[x]['id']} - {advanced_engine.agent_trajectories[x]['goal']}")
    sel_traj = advanced_engine.agent_trajectories[traj_idx]

    st.markdown(f"**Goal:** `{sel_traj['goal']}`")
    st.markdown(f"**Efficiency Score:** `{sel_traj['efficiency_score']}` | **Hallucinated Tools:** `{sel_traj['hallucinated_tools']}`")

    steps_df = pd.DataFrame(sel_traj["steps"])
    st.dataframe(steps_df, use_container_width=True)

    if sel_traj["valid_path"]:
        st.success("Trajectory Verified: Optimal tool sequence with zero schema violations.")
    else:
        st.error("Trajectory Alert: Redundant tool calls and schema errors detected in execution path.")

else:
    st.subheader("CI/CD Pull Request Quality Gate Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        base_a = st.number_input("Production Baseline Accuracy:", 0.0, 1.0, 0.92, 0.01)
    with c2:
        cand_a = st.number_input("PR Candidate Accuracy:", 0.0, 1.0, 0.89, 0.01)
    with c3:
        thresh = st.number_input("Allowable Regression Tolerance:", 0.0, 0.10, 0.02, 0.005)

    gate_res = advanced_engine.run_regression_gate(base_a, cand_a, thresh)

    st.markdown(f"**Accuracy Delta:** `{gate_res['delta']:+}` | **Tolerance Floor:** `{gate_res['threshold']}`")
    if gate_res["passed"]:
        st.success(f"VERDICT: {gate_res['verdict']} (Within acceptable tolerance band)")
    else:
        st.error(f"VERDICT: {gate_res['verdict']} (Regression exceeds {thresh:.1%} limit! Blocking CI build)")

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do you mathematically define and construct a Pareto efficiency frontier for enterprise LLM selection?"):
    st.markdown("""
    **Answer:**  
    A model is Pareto-optimal if no other model achieves strictly higher quality (e.g., accuracy, groundedness) 
    at lower or equal cost and latency. By plotting accuracy on the y-axis against cost-per-query on the x-axis, 
    the convex hull connecting the non-dominated points defines the Pareto frontier. Any model lying below 
    and to the right of the frontier is Pareto-dominated and should be decommissioned.
    """)

with st.expander("Q2: How do you design an LLM-as-a-judge prompt to guarantee deterministic, machine-parseable output?"):
    st.markdown("""
    **Answer:**  
    1. Bind a strict Pydantic V2 schema using `.with_structured_output(Schema)` or JSON schema tool calling.
    2. Enforce field ordering so the model generates step-by-step reasoning in a `chain_of_thought` string *before* generating numerical grades.
    3. Set `temperature=0.0` to eliminate sampling entropy.
    """)

with st.expander("Q3: What metrics are required to audit an agent's execution path beyond final answer correctness?"):
    st.markdown("""
    **Answer:**  
    1. **Tool Precision & Recall**: Did it select all required tools without extraneous calls?
    2. **Trajectory Efficiency (Step Ratio)**: Number of actual steps divided by the theoretical minimum path.
    3. **Argument Validity**: Schema conformance of arguments passed to tool nodes.
    4. **Loop Termination**: Ensuring the agent terminated via valid goal achievement rather than hitting recursion limits.
    """)

with st.expander("Q4: How do you prevent continuous production evaluation from exploding API costs?"):
    st.markdown("""
    **Answer:**  
    Implement stratified sampling. Evaluate a small deterministic fraction (5%–10%) of successful runs, 
    while reserving 100% evaluation for high-risk signals: user downvotes, tool errors, or outputs with 
    low confidence scores. Use lighter, cheaper judge models (e.g., GPT-4o-mini) for preliminary filtering.
    """)

with st.expander("Q5: How does LangSmith support continuous active learning from production to dataset curation?"):
    st.markdown("""
    **Answer:**  
    LangSmith allows programmatic run-to-example conversion. Using `client.create_example(inputs=run.inputs, dataset_name=...)`, 
    traces flagged with negative user feedback or low evaluator scores can be piped directly into golden triage datasets 
    for human review, prompt refinement, and regression testing.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Deploying Changes Without CI/CD Delta Assertion Gates**  
Relying on manual spot checks before merging prompt or retriever changes almost always introduces regressions. 
Automate regression gates in CI to block pull requests with negative quality deltas.
""")

st.error("""
**Trap 2: Using Unbounded Agent Recursion Without Step Penalties in Evaluators**  
An agent that takes 20 steps to answer a simple query might succeed, but costs 10x more and degrades user experience. 
Always penalize trajectory step redundancy in evaluation rubrics.
""")

st.warning("""
**Trap 3: Evaluating Production Traffic with 100% Judge Coverage**  
Running heavy LLM judges on every production query doubles or triples your cloud bill. 
Use sampling rules and reserve comprehensive grading for edge cases.
""")

st.warning("""
**Trap 4: Ignoring Model Price Updates in Pareto Analysis**  
Model providers regularly slash pricing (e.g., 50% price drops on new releases). 
Re-evaluate your Pareto cost-performance frontier quarterly.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Pareto Frontier</h4>
        <p>The set of optimal models where no alternative offers higher quality at lower cost or latency.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Trajectory Audit</h4>
        <p>Verifying the complete sequence of tool invocations, arguments, and state transitions taken by an agent.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Continuous Evaluation</h4>
        <p>The ongoing practice of sampling live production traffic to evaluate quality and backfill failure edge cases.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Delta Gate</h4>
        <p>An automated CI/CD threshold that blocks pull requests if candidate scores regress below baseline tolerance.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I can architect an automated evaluation pipeline integrated into CI/CD workflows.", value=True)
chk_2 = st.checkbox("I know how to build structured LLM-as-a-judge evaluators enforcing Pydantic schemas and Chain-of-Thought.", value=True)
chk_3 = st.checkbox("I understand how to audit multi-step agent trajectories for tool accuracy and step efficiency.", value=True)
chk_4 = st.checkbox("I can configure CI/CD regression gates using delta threshold assertions against baseline experiments.", value=True)
chk_5 = st.checkbox("I know how to perform Pareto frontier cost-performance analyses across model portfolios.", value=True)
chk_6 = st.checkbox("I understand continuous evaluation sampling and active learning dataset backfills.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced LangSmith coding and architectural competencies verified! Chapters 37 & 38 complete.")
