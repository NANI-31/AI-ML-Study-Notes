import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="22.3 Evaluator Patterns - LangGraph Curriculum",
    page_icon="⚖️",
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
    .badge-eval {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
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
class EvaluatorPatternsSimulator:
    """
    Simulates Generator-Evaluator patterns:
    - Multi-dimensional rubric scoring (Accuracy, Tone, Groundedness)
    - Weighted score calculation against quality threshold
    - Automated retry loop with temperature adjustment
    """
    def __init__(self):
        pass

    def evaluate_generation(
        self,
        task: str,
        threshold: float,
        retry_allowed: bool
    ) -> Dict[str, Any]:
        attempts = []
        passed = False
        max_attempts = 3 if retry_allowed else 1
        
        for attempt in range(1, max_attempts + 1):
            accuracy = round(0.60 + attempt * 0.12 + np.random.uniform(0.01, 0.05), 2)
            tone = round(0.70 + attempt * 0.10 + np.random.uniform(0.01, 0.04), 2)
            groundedness = round(0.65 + attempt * 0.11 + np.random.uniform(0.01, 0.05), 2)
            
            # Weighted aggregate score
            composite = round((accuracy * 0.4) + (tone * 0.2) + (groundedness * 0.4), 2)
            is_ok = composite >= threshold
            
            attempts.append({
                "attempt": attempt,
                "accuracy": accuracy,
                "tone": tone,
                "groundedness": groundedness,
                "composite_score": composite,
                "status": "PASSED" if is_ok else "RETRY_TRIGGERED"
            })
            
            if is_ok:
                passed = True
                break

        return {
            "task": task,
            "threshold": threshold,
            "passed": passed,
            "total_attempts": len(attempts),
            "attempts": attempts
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Generator-Evaluator Topologies, Rubric Scoring, and Retry Gates", divider="rainbow")

st.markdown("""
In mission-critical enterprise workflows (generating legal contracts, financial summaries, medical recommendations), delivering an unvetted model output directly to users is unacceptable.

The **Generator-Evaluator Pattern** decouples generation from quality control:
1. **Generator Node**: Generates the candidate response or artifact based on input state.
2. **Evaluator Node (LLM-as-a-Judge)**: Analyzes the artifact across multi-dimensional rubrics (accuracy, compliance, tone, groundedness) and calculates a mathematical composite score.
3. **Quality Threshold Gate**: Compares the score against an SLA threshold.
   - If `score >= threshold`: The artifact is approved and delivered.
   - If `score < threshold`: Execution branches to a **Retry Node** with specific critique notes.

The blueprint below illustrates this quality-gated architectural topology.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GeneratorEvaluatorTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_gen {
            label="1. Candidate Generation Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_ingress [label="Task Ingress\\n'Generate Non-Disclosure Agreement'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            generator_node [label="Generator Node\\nSynthesizes Candidate Contract", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_eval {
            label="2. Multi-Metric Evaluation Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            evaluator_node [label="Evaluator Node (LLM-as-a-Judge)\\nScores Accuracy, Tone, Compliance", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            score_calculator [label="Score Calculation Function\\nComposite: 0.4*Acc + 0.3*Tone + 0.3*Comp", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_gate {
            label="3. Quality Threshold & Retry Gate";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            quality_gate [label="Quality Gate\\nComposite >= 0.85 Threshold?", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            retry_node [label="Retry / Regeneration Node\\nIncreases Temp + Critique Guidance", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
            final_approved [label="Approved Deliverable\\nMeets Enterprise Compliance", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        task_ingress -> generator_node;
        generator_node -> evaluator_node [label="Candidate Draft"];
        evaluator_node -> score_calculator;
        score_calculator -> quality_gate;

        quality_gate -> final_approved [label="Score >= 0.85 (Pass)", color="#34D399"];
        quality_gate -> retry_node [label="Score < 0.85 (Fail)", color="#EF4444", style=dashed];
        retry_node -> generator_node [label="Regenerate with Notes", color="#FBBF24", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: GENERATOR
# ==============================================================================
st.header("2. Generator: Primary Synthesis and Structured Deliverable Production", divider="rainbow")

st.markdown("""
The **Generator Node** is responsible for synthesizing candidate solutions.
- **Independence**: Focuses entirely on drafting without self-evaluation overhead.
- **Output Schema**: Returns candidate content alongside generation metadata (e.g. model version, temperature used).
""")

# ==============================================================================
# SECTION 3: EVALUATOR
# ==============================================================================
st.header("3. Evaluator: LLM-as-a-Judge, Deterministic Unit Testers, and Multi-Dimensional Rubrics", divider="rainbow")

st.markdown("""
The **Evaluator Node** inspects candidate deliverables. In production systems, evaluation blends two approaches:
1. **Deterministic Evaluators**: Regex pattern matching, JSON schema validation, Pydantic type checking, unit tests.
2. **LLM-as-a-Judge**: Evaluating subjective qualities like politeness, tone, coherence, and factual groundedness against reference criteria.
""")

st.code('''from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    accuracy_score: float = Field(description="Score between 0.0 and 1.0 for technical accuracy")
    tone_score: float = Field(description="Score between 0.0 and 1.0 for executive tone adherence")
    groundedness_score: float = Field(description="Score between 0.0 and 1.0 for factual source citations")
    critique_notes: str = Field(description="Specific feedback on deficiencies")''', language="python")

# ==============================================================================
# SECTION 4: SCORE CALCULATION
# ==============================================================================
st.header("4. Score Calculation: Weighted Aggregates, Confidence Calibrations, and Grading Metrics", divider="rainbow")

st.markdown("""
Multiple rubric scores are synthesized into a single composite metric:
`Composite Score = (w_accuracy * Accuracy) + (w_tone * Tone) + (w_compliance * Compliance)`
- Ensures that failure in a non-negotiable metric (like compliance or accuracy) cannot be masked by high tone scores.
""")

# ==============================================================================
# SECTION 5: QUALITY THRESHOLD
# ==============================================================================
st.header("5. Quality Threshold: Decision Boundaries, Pass/Fail Thresholds, and SLA Compliance", divider="rainbow")

st.markdown("""
The **Quality Threshold** serves as the decision boundary in conditional edges:
- Thresholds are calibrated based on domain risk:
  - Internal Slack FAQ: `Threshold = 0.70`
  - Public Marketing Copy: `Threshold = 0.85`
  - Financial Ledger & Legal NDA: `Threshold = 0.95`
""")

# ==============================================================================
# SECTION 6: RETRY
# ==============================================================================
st.header("6. Retry: Strategic Regeneration, Parameter Perturbation, and Temperature Jitter", divider="rainbow")

st.markdown("""
When a generation fails evaluation, simply re-running the exact same prompt with the exact same temperature will likely reproduce the identical error.
Effective **Retry Strategies** include:
1. **Prompt Injection**: Appending the evaluator's critique directly into the prompt.
2. **Temperature Jitter**: Slightly increasing temperature (`0.2 -> 0.6`) to explore alternative token paths.
3. **Model Escalation**: Failing over from a smaller model (`GPT-4o-mini`) to a flagship reasoning model (`Claude 3.5 Sonnet` or `GPT-4o`).
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Generator-Evaluator Graph with Multi-Metric Quality Gates", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a Generator-Evaluator quality gate with composite scoring.
""")

st.code('''from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class GenEvalState(TypedDict):
    prompt: str
    draft: str
    composite_score: float
    critique: str
    attempts: int

def generator_node(state: GenEvalState) -> dict:
    attempt = state.get("attempts", 0) + 1
    # Generator incorporates past critique on retry
    notes = state.get("critique", "None")
    return {
        "draft": f"Generated solution v{attempt} for '{state['prompt']}' (Guided by: {notes})",
        "attempts": attempt
    }

def evaluator_node(state: GenEvalState) -> dict:
    attempt = state["attempts"]
    # Simulated multi-metric score improving with attempts
    acc = 0.70 + (attempt * 0.10)
    tone = 0.80 + (attempt * 0.05)
    composite = round((acc * 0.5) + (tone * 0.5), 2)
    critique = "Needs better compliance structure" if composite < 0.85 else "Approved"
    return {
        "composite_score": composite,
        "critique": critique
    }

def route_quality_gate(state: GenEvalState) -> str:
    # Pass if composite >= 0.85 or attempts reached max 3
    if state["composite_score"] >= 0.85 or state["attempts"] >= 3:
        return END
    return "generator"

builder = StateGraph(GenEvalState)
builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator", route_quality_gate, {"generator": "generator", END: END})

app = builder.compile(checkpointer=MemorySaver())
print("Generator-Evaluator Quality Gate Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Generator-Evaluator Sandbox & Scoring Laboratory", divider="rainbow")

st.markdown("""
Simulate multi-metric evaluation scoring across multiple attempts and observe pass/fail threshold dynamics.
""")

sim = EvaluatorPatternsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Quality Gate Controls")
    task_input = st.text_input("Generation Task", value="Formulate SaaS Service Level Agreement")
    pass_threshold = st.slider("Composite Quality Threshold (Pass/Fail)", min_value=0.70, max_value=0.95, value=0.85, step=0.05)
    allow_retries = st.toggle("Enable Automated Retries on Failure", value=True)

    if st.button("🚀 Run Generation & Evaluation"):
        res = sim.evaluate_generation(task_input, pass_threshold, allow_retries)
        st.session_state["geneval_sim_res"] = res

with col_sim2:
    st.subheader("📊 Rubric Evaluation Audit")
    if "geneval_sim_res" in st.session_state:
        res = st.session_state["geneval_sim_res"]
        
        if res["passed"]:
            st.success(f" PASSED Quality Gate in **{res['total_attempts']}** attempt(s)!")
        else:
            st.error(f"❌ FAILED Quality Gate after reaching max attempts ({res['total_attempts']}).")

        st.dataframe(pd.DataFrame(res["attempts"]), use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the primary difference between a Reflection Agent and an Evaluator Pattern?", expanded=False):
    st.markdown("""
    **Answer**:
    A Reflection Agent focuses on continuous qualitative critique and iterative textual refinement.
    The Evaluator Pattern focuses on quantitative, multi-dimensional rubric scoring against strict SLA thresholds, acting as an enterprise pass/fail quality gate.
    """)

with st.expander("Q2: Why should composite scores use weighted averages rather than a simple unweighted mean?", expanded=False):
    st.markdown("""
    **Answer**:
    Certain dimensions are non-negotiable. For example, in financial or medical systems, factual accuracy must be weighted heavily (e.g. 50-70%), whereas stylistic tone may only be weighted 10-20%.
    A weighted average ensures that flawless tone cannot compensate for factually incorrect advice.
    """)

with st.expander("Q3: How does temperature jitter help during retries?", expanded=False):
    st.markdown("""
    **Answer**:
    At temperature 0.0, an LLM is deterministic and will generate the exact same failing output if re-prompted.
    Increasing temperature slightly on retries encourages the model to explore alternate phrasing and logical paths that bypass the previous error.
    """)

with st.expander("Q4: What is LLM-as-a-Judge and what are its main failure modes?", expanded=False):
    st.markdown("""
    **Answer**:
    Using an LLM to evaluate candidate outputs against written rubrics.
    Primary failure modes include: position bias (favoring the first candidate), self-enhancement bias (favoring outputs generated by its own model family), and verbosity bias (favoring longer responses).
    """)

with st.expander("Q5: How can a production team prevent evaluation cost from exceeding generation cost?", expanded=False):
    st.markdown("""
    **Answer**:
    By using tiered evaluation: run cheap deterministic checks first (regex, Pydantic, unit tests). Only if the output passes deterministic validation should it proceed to expensive LLM-as-a-Judge evaluation.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Unbounded Retry Loops Without Max Attempt Guards")
    st.markdown("""
    *Mechanism*: Routing back to the generator purely on `score < 0.90` without checking `attempts < 3`.  
    *Consequence*: The graph cycles endlessly if the model cannot satisfy the threshold, exhausting API budgets.  
    *Fix*: Always enforce `attempts >= max_attempts` in the conditional routing edge.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Identical Re-Prompts on Retry")
    st.markdown("""
    *Mechanism*: Retrying the generator with the identical input prompt without providing critique feedback.  
    *Consequence*: The model repeats the exact same failing answer across all attempts.  
    *Fix*: Append the evaluator's critique notes directly to the retry prompt.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Verbosity Bias in LLM Evaluators")
    st.markdown("""
    *Mechanism*: Prompting an LLM judge without explicit length-neutral rubrics.  
    *Consequence*: The judge consistently awards higher scores to bloated, verbose answers over concise ones.  
    *Fix*: Explicitly penalize unnecessary fluff in the evaluator prompt rubric.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Double-LLM Calls on Every Request")
    st.markdown("""
    *Mechanism*: Running full LLM-as-a-Judge evaluations on low-stakes casual conversation.  
    *Consequence*: Doubles latency and token cost for zero business benefit.  
    *Fix*: Apply evaluators selectively to high-stakes outputs using routing gates.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Generator-Evaluator**\\nA pattern decoupling generation from independent rubric scoring and quality gates.")
with col_fc2:
    st.info("**Quality Threshold**\\nThe minimum mathematical score required for an artifact to pass evaluation.")
with col_fc3:
    st.info("**LLM-as-a-Judge**\\nUsing a secondary model to score candidate outputs against multi-dimensional rubrics.")
with col_fc4:
    st.info("**Temperature Jitter**\\nPerturbing generation parameters during retries to discover valid alternative paths.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural role of the Generator-Evaluator pattern.", value=True)
c2 = st.checkbox("I know how to design multi-dimensional rubrics using Pydantic models.", value=True)
c3 = st.checkbox("I understand how weighted composite score calculations enforce non-negotiable criteria.", value=True)
c4 = st.checkbox("I know how to implement temperature jitter and critique injection on retries.", value=True)
c5 = st.checkbox("I can build an end-to-end Generator-Evaluator quality gate in LangGraph.", value=True)
c6 = st.checkbox("I understand how to mitigate verbosity and self-enhancement biases in LLM judges.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 22.3 'Evaluator Patterns' with full architectural and practical comprehension!")
