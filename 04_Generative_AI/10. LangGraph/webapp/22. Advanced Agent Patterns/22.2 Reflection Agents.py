import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="22.2 Reflection Agents - LangGraph Curriculum",
    page_icon="🪞",
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
    .badge-reflect {
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
class ReflectionAgentsSimulator:
    """
    Simulates cyclic reflection workflows:
    - Initial candidate draft generation
    - Critique evaluation with quality scoring
    - Iterative refinement loop with stopping criteria
    """
    def __init__(self):
        pass

    def execute_reflection_loop(
        self,
        target_subject: str,
        target_quality_score: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        history = []
        current_score = 62
        current_draft = f"Initial draft on {target_subject}: Basic overview with common facts."
        
        for turn in range(1, max_iterations + 1):
            critique = f"Turn {turn} Critique: Lacks empirical metrics and formal enterprise citations." if current_score < target_quality_score else "Turn Critique: Meets all quality rubrics and guidelines."
            
            history.append({
                "turn": turn,
                "draft": current_draft,
                "score": current_score,
                "critique": critique,
                "status": "APPROVED" if current_score >= target_quality_score else "NEEDS_REVISION"
            })

            if current_score >= target_quality_score:
                break

            # Improve score on each revision
            current_score = min(current_score + int(np.random.uniform(12, 18)), 98)
            current_draft = f"Revised draft v{turn + 1} on {target_subject}: Enhanced with empirical metrics, cited sources, and formal executive structure."

        return {
            "final_draft": current_draft,
            "turns_taken": len(history),
            "converged": current_score >= target_quality_score,
            "history": history
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Cyclic Reflection and Self-Refinement Engine", divider="rainbow")

st.markdown("""
Language models frequently generate subtle errors, logical inconsistencies, or sub-optimal code on their first attempt (single-shot generation).
However, when shown their own output alongside targeted critique questions (*"Does this code handle edge cases? Are these claims supported?"*), models demonstrate remarkable ability to diagnose their own mistakes and self-correct.

The **Reflection Agent Architecture** formalizes this dialectic through a cyclic state machine:
1. **Generator Node**: Produces an initial draft or code implementation.
2. **Reflector / Critic Node**: Analyzes the draft against formal evaluation rubrics, producing explicit critique notes.
3. **Reviser Node**: Updates the draft strictly guided by the critic's recommendations.
4. **Stopping Criteria Edge**: Assesses quality score, iteration count, or convergence plateau to terminate the loop safely.

The blueprint below illustrates this cyclic self-refinement engine.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ReflectionArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_generation {
            label="1. Generation Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_input [label="Task Prompt\\n'Write a production rate-limiter in Redis'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            generator_node [label="Generator Node\\nSynthesizes Candidate Artifact", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_critique {
            label="2. Critique & Reflection Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            critic_node [label="Critic / Reflector Node\\nAudits Logic, Edge Cases, & Rubrics", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            critique_memo [label="Critique Channel State\\n'Race condition in Lua script; missing TTL'", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_revision_stop {
            label="3. Revision & Stopping Criteria Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            reviser_node [label="Reviser Node\\nApplies Targeted Refinements", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            stop_gate [label="Stopping Criteria Gate\\nScore >= 90 OR Iterations >= 3?", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            final_delivery [label="Refined Production Artifact\\nZero Flaws Verified", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        user_input -> generator_node;
        generator_node -> critic_node [label="Initial Draft"];
        critic_node -> critique_memo;
        critique_memo -> stop_gate;

        stop_gate -> reviser_node [label="Quality Below Threshold (Loop)", color="#EF4444", style=dashed];
        reviser_node -> critic_node [label="Revised Draft"];
        stop_gate -> final_delivery [label="Quality Satisfied (Exit)", color="#34D399"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: GENERATE
# ==============================================================================
st.header("2. Generate: Producing the Initial Candidate Draft or Code Implementation", divider="rainbow")

st.markdown("""
The **Generator Node** takes the user's objective and produces candidate v1.
- Focus: Rapid initial solution generation without over-thinking every edge case.
- Output: Stored in an artifact channel (e.g. `current_draft` or `code_snippet`).
""")

# ==============================================================================
# SECTION 3: CRITIQUE
# ==============================================================================
st.header("3. Critique: Objective Evaluation Against Rubrics, Guidelines, and Test Cases", divider="rainbow")

st.markdown("""
The **Critic Node** evaluates the candidate output against an explicit rubric:
- Evaluates clarity, correctness, edge case handling, and security.
- **Rule of Thumb**: The critique must be **actionable**. A critique saying *"this is bad"* is useless; it must specify *"line 14 lacks a null check for user_id"*.
""")

# ==============================================================================
# SECTION 4: REVISE
# ==============================================================================
st.header("4. Revise: Targeted Refinement Driven by Actionable Critique Feedback", divider="rainbow")

st.markdown("""
The **Reviser Node** consumes:
1. The original goal.
2. The current draft.
3. The specific critique notes.

It edits only the identified deficiencies, preserving the strengths of the previous version.
""")

# ==============================================================================
# SECTION 5: REPEAT
# ==============================================================================
st.header("5. Repeat: Multi-Turn Convergence Cycles and State History Tracking", divider="rainbow")

st.markdown("""
Reflection operates as a **cyclic loop**.
Each iteration increases artifact quality:
- Iteration 1: Solves core logic (Score: 65%).
- Iteration 2: Fixes edge cases and performance bottlenecks (Score: 85%).
- Iteration 3: Polishes documentation and formatting (Score: 95%).
""")

# ==============================================================================
# SECTION 6: STOP CRITERIA
# ==============================================================================
st.header("6. Stop Criteria: Quality Thresholds, Maximum Iteration Guards, and Plateau Detection", divider="rainbow")

st.markdown("""
To prevent endless cycling, production graphs enforce **three stopping criteria**:
1. **Quality Threshold Met**: The critic gives a score above the target SLA (e.g. `score >= 90`).
2. **Maximum Iterations Reached**: Hard ceiling on loop count (`iteration >= 3`).
3. **Plateau Detection**: If the score does not improve between two consecutive revisions, terminate early.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Multi-Turn Reflection and Self-Correction Graph", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a cyclic Reflection graph in LangGraph with quality score thresholding.
""")

st.code('''from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ReflectionState(TypedDict):
    topic: str
    draft: str
    critique: str
    quality_score: int
    iteration: int

def generate_node(state: ReflectionState) -> dict:
    return {
        "draft": f"Initial implementation for: {state['topic']}",
        "iteration": 1
    }

def critic_node(state: ReflectionState) -> dict:
    iter_num = state["iteration"]
    # Simulated critic scoring: improves with each iteration
    score = 65 + (iter_num * 15)
    critique = f"Iteration {iter_num}: Missing edge case handling." if score < 90 else "Approved: Excellent quality."
    return {
        "quality_score": score,
        "critique": critique
    }

def revise_node(state: ReflectionState) -> dict:
    return {
        "draft": f"Refined implementation v{state['iteration'] + 1} incorporating: {state['critique']}",
        "iteration": state["iteration"] + 1
    }

def route_reflection(state: ReflectionState) -> str:
    # Stop condition: score >= 90 OR iterations >= 3
    if state["quality_score"] >= 90 or state["iteration"] >= 3:
        return END
    return "revise"

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_node)
builder.add_node("critic", critic_node)
builder.add_node("revise", revise_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critic")
builder.add_conditional_edges("critic", route_reflection, {"revise": "revise", END: END})
builder.add_edge("revise", "critic")

app = builder.compile(checkpointer=MemorySaver())
print("Reflection Cyclic Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Reflection & Critique Loop Laboratory", divider="rainbow")

st.markdown("""
Observe how iterative reflection and critique loops drive convergence toward target quality scores.
""")

sim = ReflectionAgentsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Reflection Configuration")
    target_subject = st.text_input("Artifact Objective", value="Design Token Bucket Rate Limiter")
    target_score = st.slider("Target Quality Score (Exit Threshold)", min_value=70, max_value=95, value=85, step=5)
    max_loops = st.slider("Maximum Iteration Guardrail", min_value=1, max_value=5, value=3)

    if st.button("🚀 Run Reflection Loop"):
        ref_res = sim.execute_reflection_loop(target_subject, target_score, max_loops)
        st.session_state["ref_sim_res"] = ref_res

with col_sim2:
    st.subheader("📋 Convergence Audit & History")
    if "ref_sim_res" in st.session_state:
        res = st.session_state["ref_sim_res"]
        
        if res["converged"]:
            st.success(f" Converged in **{res['turns_taken']}** turns!")
        else:
            st.warning(f"⚠️ Reached max loop limit ({res['turns_taken']}) before hitting target score.")

        st.dataframe(pd.DataFrame(res["history"])[["turn", "score", "status", "critique"]], use_container_width=True)
        st.markdown("#### Final Refined Deliverable:")
        st.code(res["final_draft"])

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the core mechanism of a Reflection Agent?", expanded=False):
    st.markdown("""
    **Answer**:
    A cyclic feedback loop where a generator produces a candidate draft, a critic evaluates the draft against rubrics and produces targeted feedback, and a reviser refines the draft until stopping criteria are satisfied.
    """)

with st.expander("Q2: Why should the Generator and Critic use different system prompts?", expanded=False):
    st.markdown("""
    **Answer**:
    To eliminate confirmation bias. If an agent reviews its own work with the same creative framing, it will overlook its own assumptions. A dedicated critic prompt adopts an adversarial, skeptical perspective.
    """)

with st.expander("Q3: What are the three essential stopping criteria in reflection graphs?", expanded=False):
    st.markdown("""
    **Answer**:
    1. Score threshold met (`score >= target`).
    2. Maximum loop iteration count (`iteration >= max_limit`).
    3. Performance plateau detection (score delta between consecutive turns is zero).
    """)

with st.expander("Q4: Can reflection agents verify code using unit tests?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Instead of only using an LLM critic, a code reflection graph runs the generated code against actual unit tests in a Python REPL. The compiler output and stack traces are provided as the critique.
    """)

with st.expander("Q5: What is the latency trade-off of reflection agents?", expanded=False):
    st.markdown("""
    **Answer**:
    Reflection agents trade latency for quality. Each revision loop requires 2 additional LLM calls (critic + reviser), multiplying response time by 2x to 4x.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Reflection Loops Without Max Iterations")
    st.markdown("""
    *Mechanism*: Defining `builder.add_conditional_edges("critic", route_fn)` with only `score >= 90` as the exit condition.  
    *Consequence*: If the model plateaus at 88%, it loops forever until hitting recursion limits.  
    *Fix*: Always combine quality thresholds with `iteration >= max_turns`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Over-Critiquing Causing Hallucinated Regression")
    st.markdown("""
    *Mechanism*: A critic that is too harsh, forcing the reviser to change already-correct code.  
    *Consequence*: The reviser introduces new bugs in turn 3 that did not exist in turn 1.  
    *Fix*: Instruct the critic to evaluate only against explicit checklist items.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Wiping State History Between Revision Turns")
    st.markdown("""
    *Mechanism*: Not preserving past critiques in state.  
    *Consequence*: The reviser undoes a fix from turn 1 when trying to solve a critique in turn 2.  
    *Fix*: Maintain an append-only `critique_history` list.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Cost Warning): Running Reflection on Real-Time Conversational Chat")
    st.markdown("""
    *Mechanism*: Running 3-turn reflection loops on every casual customer message.  
    *Consequence*: Multiplies token costs 4x and delays responses by 6+ seconds.  
    *Fix*: Reserve reflection graphs for offline documents, code generation, and complex reports.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Reflection Loop**\\nA cyclic state machine alternating between generation, critique, and revision.")
with col_fc2:
    st.info("**Critic Node**\\nAn independent evaluator auditing drafts against objective quality rubrics.")
with col_fc3:
    st.info("**Actionable Critique**\\nSpecific, concrete revision recommendations rather than vague quality judgements.")
with col_fc4:
    st.info("**Stopping Criteria**\\nConditions (threshold, max loops, plateau) that safely terminate the cycle.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the cyclic architecture of Reflection Agents.", value=True)
c2 = st.checkbox("I know why the Critic and Generator require distinct personas.", value=True)
c3 = st.checkbox("I understand all 3 core stopping criteria for terminating reflection loops.", value=True)
c4 = st.checkbox("I know how to implement deterministic unit-test feedback in code reflection loops.", value=True)
c5 = st.checkbox("I can implement a complete Reflection state machine in LangGraph.", value=True)
c6 = st.checkbox("I understand the latency and cost trade-offs of multi-turn reflection.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 22.2 'Reflection Agents' with full architectural and practical comprehension!")
