import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="22.4 Advanced Reasoning Workflows - LangGraph Curriculum",
    page_icon="🧠",
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
    .badge-reasoning {
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
class AdvancedReasoningSimulator:
    """
    Simulates advanced reasoning topologies:
    - Ensemble generation across 3 diverse model outputs
    - Majority voting & consensus resolution
    - Tree-of-Thoughts exploration with branch pruning
    """
    def __init__(self):
        pass

    def run_ensemble_voting(self, prompt: str) -> Dict[str, Any]:
        candidates = [
            {"model": "Model A (GPT-4o)", "predicted_answer": "Option B (Linear Complexity)", "confidence": 0.94},
            {"model": "Model B (Claude 3.5)", "predicted_answer": "Option B (Linear Complexity)", "confidence": 0.91},
            {"model": "Model C (Llama 3.3)", "predicted_answer": "Option A (Logarithmic)", "confidence": 0.62}
        ]
        
        # Calculate vote tallies
        tallies = {}
        for c in candidates:
            ans = c["predicted_answer"]
            tallies[ans] = tallies.get(ans, 0) + 1
        
        winner = max(tallies, key=tallies.get)
        agreement_ratio = tallies[winner] / len(candidates)

        return {
            "prompt": prompt,
            "candidates": candidates,
            "tallies": tallies,
            "winner": winner,
            "agreement_ratio": agreement_ratio
        }

    def run_tree_of_thoughts(self, problem: str, branching_factor: int) -> Dict[str, Any]:
        tree = []
        for b in range(1, branching_factor + 1):
            eval_score = round(0.50 + b * 0.15 + np.random.uniform(0.01, 0.05), 2)
            tree.append({
                "branch_id": f"Thought_Path_{b}",
                "hypothesis": f"Exploration path {b} for '{problem}'",
                "heuristic_score": eval_score,
                "action": "PRUNED" if eval_score < 0.70 else "EXPANDED"
            })
        best_path = max(tree, key=lambda x: x["heuristic_score"])
        return {
            "problem": problem,
            "tree": tree,
            "best_path": best_path
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Deliberative Reasoning Topologies in LangGraph", divider="rainbow")

st.markdown("""
For complex mathematical, analytical, or strategic problems, standard single-path generation is insufficient.
Advanced agent engineering employs **deliberative reasoning workflows** inspired by human cognitive science:
- **Self-Correction & Self-Verification**: Automated recovery from compiler errors and explicit fact-checking assertions.
- **Debate & Dialectics**: Pitting diverse perspectives against each other to surface hidden assumptions.
- **Voting & Ensemble Reasoning**: Querying multiple distinct model architectures in parallel and taking the majority or weighted consensus.
- **Tree-Style Exploration (Tree of Thoughts)**: Branching candidate hypotheses, scoring them with heuristic evaluators, and pruning unpromising branches.
- **Iterative Reasoning**: Multi-step convergence loops that systematically refine intermediate representations.

The blueprint below illustrates these advanced deliberative reasoning topologies.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedReasoningTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_ensemble {
            label="1. Ensemble Generation Tier (Parallel Fan-Out)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_input [label="Complex Reasoning Task\\n(Mathematical / System Design)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            model_a [label="Model A (GPT-4o)\\nGenerates Candidate 1", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            model_b [label="Model B (Claude 3.5)\\nGenerates Candidate 2", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            model_c [label="Model C (DeepSeek-R1)\\nGenerates Candidate 3", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            task_input -> model_a;
            task_input -> model_b;
            task_input -> model_c;
        }

        subgraph cluster_consensus {
            label="2. Consensus & Voting Gate (Aggregation Tier)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            voting_node [label="Majority Voting & Judge Node\\nTallies Agreement & Discrepancies", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            consensus_gate [label="Consensus Gate\\nAgreement >= 66%?", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            model_a -> voting_node;
            model_b -> voting_node;
            model_c -> voting_node;
            voting_node -> consensus_gate;
        }

        subgraph cluster_resolution {
            label="3. Resolution Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            tot_fallback [label="Tree-of-Thoughts Exploration\\nHeuristic Search & Branch Pruning", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
            final_approved [label="Consensus Verified Delivery\\nHigh Confidence Result", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            consensus_gate -> final_approved [label="Consensus Reached", color="#34D399"];
            consensus_gate -> tot_fallback [label="Split Decision (Tree Search)", color="#EF4444", style=dashed];
            tot_fallback -> final_approved [label="Optimal Branch Found", color="#38BDF8"];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: SELF-CORRECTION
# ==============================================================================
st.header("2. Self-Correction: Runtime Error Recovery, Stack Trace Feeding, and Automated Repair", divider="rainbow")

st.markdown("""
**Self-Correction** is the ability of an agent to recognize when an execution failed and autonomously fix its implementation.
- **Traceback Ingestion**: In coding agents, runtime errors (e.g., `IndexError`, `SyntaxError`) are formatted directly into the state channel as feedback.
- **Root Cause Analysis**: The model is prompted with: *"Your previous code raised the following traceback. Identify the flaw and regenerate."*
- **Empirical Gain**: Self-correction increases code generation success rates from ~60% to over 85% within 2 iterations.
""")

# ==============================================================================
# SECTION 3: SELF-VERIFICATION
# ==============================================================================
st.header("3. Self-Verification: Factual Grounding Verification and Guardrail Assertion Checks", divider="rainbow")

st.markdown("""
Unlike self-correction (which responds to explicit runtime crashes), **Self-Verification** checks whether logical conclusions are valid even when no crash occurred.
- Formulates verification questions: *"Does Step 3 logically follow from Step 2?"*
- Verifies mathematical calculations by recalculating step-by-step in reverse.
- Ensures all factual claims are backed by citations in the context state.
""")

# ==============================================================================
# SECTION 4: DEBATE
# ==============================================================================
st.header("4. Debate: Multi-Agent Dialectic and Adversarial Exploration", divider="rainbow")

st.markdown("""
The **Debate Pattern** engages two opposing agent personas:
- **Advocate (Thesis)**: Defends a policy, recommendation, or architectural choice.
- **Critic (Antithesis)**: Attacks vulnerabilities, identifies risks, and highlights edge cases.
- **Outcome**: A neutral Arbiter synthesizes the arguments into a calibrated compromise, avoiding single-model groupthink.
""")

# ==============================================================================
# SECTION 5: VOTING
# ==============================================================================
st.header("5. Voting: Majority Voting, Weighted Consensus, and Tie-Breaking Protocols", divider="rainbow")

st.markdown("""
In **Voting**, multiple candidate solutions are generated (either by sampling the same model at `temperature=0.7` or by querying diverse foundation models).
- **Plurality Voting**: The most frequently generated exact answer wins.
- **Weighted Voting**: Votes are weighted by each model's benchmark performance or self-reported confidence.
- **Tie-Breaking**: If votes are split, a senior judge node reviews the reasoning traces to select the superior solution.
""")

# ==============================================================================
# SECTION 6: ENSEMBLE REASONING
# ==============================================================================
st.header("6. Ensemble Reasoning: Multi-Model Querying and Diversity-Driven Aggregation", divider="rainbow")

st.markdown("""
Different model families possess distinct training biases, strengths, and failure modes:
- Model A might excel at creative synthesis but make minor arithmetic mistakes.
- Model B might excel at rigorous symbolic logic.
- Model C might have superior retrieval recall.

**Ensemble Reasoning** queries diverse models in parallel and aggregates their outputs into a single consolidated deliverable.
""")

# ==============================================================================
# SECTION 7: TREE-STYLE EXPLORATION
# ==============================================================================
st.header("7. Tree-Style Exploration: Tree-of-Thoughts (ToT), Branching Heuristics, and Backtracking", divider="rainbow")

st.markdown("""
Standard Chain-of-Thought (CoT) follows a linear reasoning path. If an early step is flawed, the entire chain fails.
**Tree of Thoughts (ToT)** explores a tree of possibilities:
1. **Thought Generation**: Proposes $k$ alternative hypotheses for the next step.
2. **State Evaluation**: Evaluates each branch using a heuristic scoring function.
3. **Search Algorithm**: Uses Depth-First Search (DFS) or Breadth-First Search (BFS) to expand promising branches and **prune** unpromising ones.
""")

# ==============================================================================
# SECTION 8: ITERATIVE REASONING
# ==============================================================================
st.header("8. Iterative Reasoning: Stepwise Refinement and Multi-Stage Convergence", divider="rainbow")

st.markdown("""
**Iterative Reasoning** structures problem-solving across explicit deliberative stages:
`Formulate Hypotheses -> Design Tests -> Simulate Outcomes -> Select Best Strategy`.
Each stage updates the LangGraph state, providing complete auditability at every step.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Model Ensemble and Majority Voting Graph", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing parallel ensemble generation and majority voting in LangGraph.
""")

st.code('''from typing import TypedDict, List
from collections import Counter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class EnsembleState(TypedDict):
    prompt: str
    candidates: List[str]
    winning_answer: str
    agreement_percent: float

def model_a_node(state: EnsembleState) -> dict:
    # Simulates Model A output
    return {"candidates": state.get("candidates", []) + ["Answer: O(N) Linear"]}

def model_b_node(state: EnsembleState) -> dict:
    # Simulates Model B output
    return {"candidates": state.get("candidates", []) + ["Answer: O(N) Linear"]}

def model_c_node(state: EnsembleState) -> dict:
    # Simulates Model C output
    return {"candidates": state.get("candidates", []) + ["Answer: O(log N) Logarithmic"]}

def voting_judge_node(state: EnsembleState) -> dict:
    candidates = state["candidates"]
    counts = Counter(candidates)
    winner, top_votes = counts.most_common(1)[0]
    agreement = (top_votes / len(candidates)) * 100.0
    return {
        "winning_answer": winner,
        "agreement_percent": agreement
    }

builder = StateGraph(EnsembleState)
builder.add_node("model_a", model_a_node)
builder.add_node("model_b", model_b_node)
builder.add_node("model_c", model_c_node)
builder.add_node("voting_judge", voting_judge_node)

# Execute models sequentially or concurrently
builder.add_edge(START, "model_a")
builder.add_edge("model_a", "model_b")
builder.add_edge("model_b", "model_c")
builder.add_edge("model_c", "voting_judge")
builder.add_edge("voting_judge", END)

app = builder.compile(checkpointer=MemorySaver())
print("Ensemble Majority Voting Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Advanced Reasoning & Voting Simulator", divider="rainbow")

st.markdown("""
Simulate multi-model ensemble voting and explore Tree-of-Thoughts heuristic branch pruning.
""")

sim = AdvancedReasoningSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("🗳️ Simulation 1: Multi-Model Ensemble Voting")
    test_task = st.text_input("Reasoning Problem", value="Determine Time Complexity of Sliding Window Algorithm")
    
    if st.button("🚀 Execute Ensemble Voting"):
        vote_res = sim.run_ensemble_voting(test_task)
        st.success(f"Consensus Winner: **{vote_res['winner']}** (Agreement: {int(vote_res['agreement_ratio']*100)}%)")
        st.dataframe(pd.DataFrame(vote_res["candidates"]), use_container_width=True)

with col_sim2:
    st.subheader("🌲 Simulation 2: Tree-of-Thoughts Branch Pruning")
    problem_tot = st.text_input("Complex Theorem Goal", value="Optimize Global Supply Chain Route")
    branches = st.slider("Branching Factor (Candidate Paths)", min_value=2, max_value=5, value=3)

    if st.button("⚡ Explore Reasoning Tree"):
        tot_res = sim.run_tree_of_thoughts(problem_tot, branches)
        st.info(f"Optimal Path Selected: **{tot_res['best_path']['branch_id']}** (Score: {tot_res['best_path']['heuristic_score']})")
        st.dataframe(pd.DataFrame(tot_res["tree"]), use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the Tree of Thoughts (ToT) framework and how is it implemented in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    ToT generalizes Chain-of-Thought by exploring multiple reasoning paths organized as a tree.
    In LangGraph, ToT is implemented using the `Send` API or parallel nodes to generate candidate thoughts, an evaluator node to score each path with heuristics, and conditional edges that prune low-scoring branches and expand promising ones.
    """)

with st.expander("Q2: Why does multi-model ensemble voting improve accuracy on difficult reasoning tasks?", expanded=False):
    st.markdown("""
    **Answer**:
    Different model architectures (e.g. OpenAI, Anthropic, open-source) have independent pre-training distributions and failure modes.
    Ensemble voting averages out individual model hallucinations; if 2 out of 3 independent models reach the exact same conclusion, the probability of correctness is significantly higher.
    """)

with st.expander("Q3: How does Self-Verification differ from simple generation?", expanded=False):
    st.markdown("""
    **Answer**:
    Simple generation generates forward tokens without checking logic.
    Self-Verification adds an explicit validation phase where the model generates verification questions, evaluates its own intermediate claims against ground truth, and asserts consistency.
    """)

with st.expander("Q4: What is the primary bottleneck when implementing Tree of Thoughts in production?", expanded=False):
    st.markdown("""
    **Answer**:
    Exponential branching complexity ($O(b^d)$) and token latency.
    Without aggressive heuristic pruning at depth 1 and 2, the number of LLM calls explodes rapidly, making the system slow and expensive.
    """)

with st.expander("Q5: How does LangGraph handle tie-breaking in ensemble voting workflows?", expanded=False):
    st.markdown("""
    **Answer**:
    Using a dedicated Arbiter or Judge node. When vote counts are tied, execution routes to the Arbiter, which reads the reasoning traces of all candidates and selects the winning solution based on argument strength.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Unbounded Tree Branching Causing State Explosion")
    st.markdown("""
    *Mechanism*: Branching 5 thoughts at depth 4 without pruning ($5^4 = 625$ nodes).  
    *Consequence*: Hundreds of concurrent LLM calls exhaust rate limits and crash the graph.  
    *Fix*: Restrict branching to $b \\le 3$, depth $\\le 2$, and prune aggressively.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Correlated Failures in Ensemble Voting")
    st.markdown("""
    *Mechanism*: Ensembling 3 model checkpoints that are all fine-tunes of the identical base model.  
    *Consequence*: All 3 models share the exact same training blind spots and vote together on the wrong answer.  
    *Fix*: Use genuinely diverse model families (e.g. OpenAI + Anthropic + DeepSeek).
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Cost Warning): Running Ensembles on Deterministic Tasks")
    st.markdown("""
    *Mechanism*: Running 3 models and a voting judge for a simple classification task.  
    *Consequence*: Multiplies inference cost by 4x for marginal accuracy gain.  
    *Fix*: Reserve ensemble voting for high-value, highly ambiguous queries.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Synchronous Sequential Ensemble Execution")
    st.markdown("""
    *Mechanism*: Running Model A, waiting 2s, running Model B, waiting 2s, running Model C.  
    *Consequence*: Latency equals the sum of all three models (6s).  
    *Fix*: Execute ensemble models in parallel using LangGraph parallel branches or `asyncio.gather`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Tree of Thoughts**\\nHeuristic search exploring branching candidate hypotheses with backtracking.")
with col_fc2:
    st.info("**Ensemble Voting**\\nQuerying diverse models in parallel and resolving answers by majority consensus.")
with col_fc3:
    st.info("**Self-Correction**\\nFeeding runtime error traces back into the generator to trigger automated repairs.")
with col_fc4:
    st.info("**Branch Pruning**\\nTerminating unpromising hypothesis paths early to prevent exponential token explosion.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the mechanics of Self-Correction and stack trace feeding.", value=True)
c2 = st.checkbox("I know how to implement parallel Ensemble Generation across diverse models.", value=True)
c3 = st.checkbox("I understand majority voting, weighted consensus, and arbiter tie-breaking.", value=True)
c4 = st.checkbox("I understand the Tree-of-Thoughts framework and branch pruning heuristics.", value=True)
c5 = st.checkbox("I can implement an end-to-end Ensemble Majority Voting graph in LangGraph.", value=True)
c6 = st.checkbox("I know how to avoid exponential state explosion during tree-style exploration.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 22.4 'Advanced Reasoning Workflows' with full architectural and practical comprehension!")
