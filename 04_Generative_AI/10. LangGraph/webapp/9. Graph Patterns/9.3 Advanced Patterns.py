import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="9.3 Advanced Patterns - LangGraph Curriculum",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE ADVERSARIAL DEBATE SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class AdvancedPatternsEngine:
    """
    Simulates advanced multi-agent collective intelligence:
    Adversarial Multi-Agent Debate (Proponent vs Opponent -> Judge Verdict).
    """
    def __init__(self):
        self.advanced_patterns = {
            "Reflection": "Agent pauses to critique its own reasoning against an evaluation rubric.",
            "Self-Correction": "External test harness provides execution tracebacks to fix bugs.",
            "Iterative Refinement": "Progressive multi-round improvement towards quality convergence.",
            "Debate": "Two adversarial agents challenge each other's premises before a neutral judge.",
            "Ensemble": "Multiple diverse models run concurrently to mitigate single-model bias.",
            "Voting": "Aggregates categorical predictions across multiple agents via majority rule.",
            "Hierarchical Workflows": "Multi-tier supervisor delegating sub-tasks to encapsulated subgraphs."
        }

    def simulate_debate(self, proposition: str, rounds: int) -> Dict[str, Any]:
        """
        Simulates a multi-turn debate between Agent Pro and Agent Con, followed by a Judge.
        """
        debate_turns = []
        for r in range(1, rounds + 1):
            debate_turns.append({
                "round": r,
                "agent": "Proponent Agent (Affirmative)",
                "argument": f"Round {r}: Empirical evidence supports '{proposition}'. Distributed checkpointers eliminate single points of failure."
            })
            debate_turns.append({
                "round": r,
                "agent": "Opponent Agent (Negative)",
                "argument": f"Round {r}: Counter-evidence reveals operational complexity and checkpointer I/O overhead for '{proposition}'."
            })

        verdict = f"Judge Verdict: After {rounds} rounds of dialectic debate on '{proposition}', the Proponent provided superior empirical benchmarks. Approved with mitigation guidelines."
        return {
            "proposition": proposition,
            "turns": debate_turns,
            "verdict": verdict,
            "confidence_score": 0.88
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Multi-Agent Collective Intelligence Patterns", divider="rainbow")

st.markdown("""
When tackling high-stakes decision making, single-agent architectures are limited by the inherent cognitive biases
of an individual prompt or model. **Advanced Collective Intelligence Patterns** leverage multi-agent interaction:
**Adversarial Debate**, **Hierarchical Supervision**, **Ensemble Divergence**, and **Consensus Voting**.
LangGraph orchestrates these multi-agent graphs with persistent state, cyclical dialogue turns, and isolated subgraphs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    debate_blueprint_dot = """
    digraph AdvancedDebateArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress & Topic Framing";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nProposition or Policy Question", fillcolor="#1E293B", shape=ellipse];
            Moderator [label="Moderator / Supervisor Node\\nFrames debate rules and tracks round counter", fillcolor="#0284C7"];
        }

        subgraph cluster_adversarial_core {
            label="2. Adversarial Dialectic Core (Cyclic Debate)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Agent_Pro [label="Proponent Agent (Affirmative)\\nConstructs supporting arguments", fillcolor="#4338CA"];
            Agent_Con [label="Opponent Agent (Negative)\\nIdentifies flaws, blindspots, and risks", fillcolor="#B91C1C"];
            Round_Gate [label="Debate Finished?\\nrounds >= limit?", shape=diamond, fillcolor="#6D28D9"];
            
            Agent_Pro -> Agent_Con [label="Presents argument"];
            Agent_Con -> Round_Gate [label="Rebuttal submitted"];
            Round_Gate -> Agent_Pro [label="Next Round (rounds < limit)"];
        }

        subgraph cluster_arbitration {
            label="3. Neutral Arbitration (Reduce & Verdict)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Judge_Node [label="Neutral Chief Justice / Judge Node\\nEvaluates argument strength, facts, and logical fallacies", fillcolor="#059669"];
            END [label="END\\nFinal Balanced Policy Delivered", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Moderator;
        Moderator -> Agent_Pro [label="Open debate"];
        Round_Gate -> Judge_Node [label="Debate Concluded"];
        Judge_Node -> END;
    }
    """
    st.graphviz_chart(debate_blueprint_dot, use_container_width=True)

st.markdown("""
### Dialectic Reasoning vs Single-Model Hallucination
Adversarial debate mirrors the scientific and legal methods: by forcing two distinct agent personas
to rigorously critique each other's claims, hallucinations are aggressively challenged before reaching the judge.
""")

# ==============================================================================
# SECTION 2: REFLECTION
# ==============================================================================
st.header("2. Reflection: Internal Dialogue and Evaluator-Optimizer Introspection", divider="rainbow")

st.markdown("""
**Reflection** is the introspective pattern where an agent evaluates its own prior thoughts before committing an action.
In LangGraph, reflection is formalized as a cyclic state machine:
`Node_Generate -> Node_Reflect -> Router(Pass / Try Again)`.
""")

# ==============================================================================
# SECTION 3: SELF-CORRECTION
# ==============================================================================
st.header("3. Self-Correction: Grounded Unit-Test Verification and Automated Bug Patching", divider="rainbow")

st.markdown("""
**Self-Correction** grounds reasoning in deterministic verification harnesses (code compilers, JSON schema validators, unit tests).
When verification fails, the compiler error output is injected into state as the next prompt's user input.
""")

# ==============================================================================
# SECTION 4: ITERATIVE REFINEMENT
# ==============================================================================
st.header("4. Iterative Refinement: Convergence Invariants and Monotonic Quality Progression", divider="rainbow")

st.markdown("""
**Iterative Refinement** executes continuous drafting cycles.
To ensure mathematical convergence:
1. State tracks `best_draft` and `best_score`.
2. Revisions that score lower than the current best draft are discarded.
3. The process terminates when score delta drops below a threshold (delta < epsilon).
""")

# ==============================================================================
# SECTION 5: DEBATE
# ==============================================================================
st.header("5. Debate: Multi-Agent Dialectic and Adversarial Cross-Examination", divider="rainbow")

st.markdown("""
In the **Debate Pattern**, two or more agents with opposing system prompts (e.g. Bull vs Bear, Prosecutor vs Defense)
engage in structured conversational turns. Neither agent determines the final outcome;
a neutral **Judge Node** reviews the complete dialogue transcript to render an impartial verdict.
""")

# ==============================================================================
# SECTION 6: ENSEMBLE
# ==============================================================================
st.header("6. Ensemble: Diverse Model Sampling and Heterogeneous Prompt Architectures", divider="rainbow")

st.markdown("""
An **Ensemble Pattern** runs multiple heterogeneous models (e.g. Claude 3.5 Sonnet, GPT-4o, and Gemini 1.5 Pro)
in parallel via `Send`. Because each model family possesses different training corpora and bias profiles,
ensemble architectures dramatically diminish single-provider vulnerabilities.
""")

# ==============================================================================
# SECTION 7: VOTING
# ==============================================================================
st.header("7. Voting: Majority Consensus, Borda Count, and Judge-Mediated Arbitration", divider="rainbow")

st.markdown("""
**Voting Mechanisms** consolidate predictions across multiple agents:
- **Hard Majority**: Candidate with > 50% votes wins.
- **Plurality Voting**: Candidate with the highest vote count wins.
- **Borda Count**: Agents submit ranked preferences (1st, 2nd, 3rd) with points assigned per rank.
""")

# ==============================================================================
# SECTION 8: HIERARCHICAL WORKFLOWS
# ==============================================================================
st.header("8. Hierarchical Workflows: Supervisor Delegations and Subgraph Encapsulation", divider="rainbow")

st.markdown("""
In **Hierarchical Workflows**, a **Supervisor Node** coordinates specialized teams.
Each team is encapsulated as an independent `CompiledStateGraph` with its own private memory schema.
The supervisor only receives high-level status reports, enabling massive organizational complexity.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Agent Adversarial Debate Pipeline", divider="rainbow")

st.markdown("""
Below is a complete production implementation of a Multi-Agent Adversarial Debate pipeline in LangGraph.
Proponent and Opponent alternate argument rounds, and an impartial Judge resolves the debate.
""")

prod_debate_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Step 1: Define Debate State
class DebateState(TypedDict):
    proposition: str
    round_count: int
    max_rounds: int
    transcript: Annotated[List[str], operator.add]
    final_verdict: str

# Nodes
def proponent_node(state: DebateState) -> Dict[str, Any]:
    r = state["round_count"] + 1
    arg = f"Round {r} Affirmative: Distributed checkpointers guarantee resilient fault recovery."
    return {
        "round_count": r,
        "transcript": [arg]
    }

def opponent_node(state: DebateState) -> Dict[str, Any]:
    r = state["round_count"]
    rebuttal = f"Round {r} Negative: However, checkpointer I/O introduces serialization latency overhead."
    return {
        "transcript": [rebuttal]
    }

def judge_node(state: DebateState) -> Dict[str, Any]:
    transcript = state["transcript"]
    verdict = f"Chief Justice: Evaluated {len(transcript)} arguments. Affirmative wins due to verifiable durability guarantees."
    return {"final_verdict": verdict}

# Router Predicate
def should_continue_debate(state: DebateState) -> str:
    if state["round_count"] < state["max_rounds"]:
        return "proponent"
    return "judge"

# Step 2: Build Graph
def create_debate_graph():
    builder = StateGraph(DebateState)
    
    builder.add_node("proponent", proponent_node)
    builder.add_node("opponent", opponent_node)
    builder.add_node("judge", judge_node)
    
    builder.add_edge(START, "proponent")
    builder.add_edge("proponent", "opponent")
    
    builder.add_conditional_edges(
        "opponent",
        should_continue_debate,
        {"proponent": "proponent", "judge": "judge"}
    )
    builder.add_edge("judge", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = create_debate_graph()
    initial_debate = {
        "proposition": "Should enterprise microservices adopt cyclic state machines?",
        "round_count": 0,
        "max_rounds": 2,
        "transcript": [],
        "final_verdict": ""
    }
    result = app.invoke(initial_debate)
    print("Verdict:", result["final_verdict"])
    for line in result["transcript"]:
        print("  -", line)
'''
st.code(prod_debate_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Multi-Agent Debate and Consensus Simulation Engine", divider="rainbow")

st.markdown("""
Simulate an adversarial multi-agent debate in real-time. Enter a technical proposition,
set the number of debate rounds, and observe Proponent and Opponent spar before the Judge.
""")

engine = AdvancedPatternsEngine()

col_dw_l, col_dw_r = st.columns([1, 1.2])

with col_dw_l:
    st.subheader("Configure Multi-Agent Debate")
    test_proposition = st.text_input(
        "Proposition to Debate",
        value="Postgres checkpointer is strictly superior to Redis for agent durability"
    )
    rounds_count = st.slider("Debate Rounds", 1, 3, 2)

    if st.button("⚖️ Initiate Multi-Agent Dialectic Debate", use_container_width=True):
        st.session_state.debate_run = engine.simulate_debate(test_proposition, rounds_count)
        st.success("Debate completed and verdict rendered!")

with col_dw_r:
    st.subheader("Transcript & Verdict")
    if "debate_run" in st.session_state:
        run = st.session_state.debate_run
        st.success(run["verdict"])
        
        st.markdown("**Debate Transcript:**")
        for turn in run["turns"]:
            with st.expander(f"Round {turn['round']}: {turn['agent']}", expanded=True):
                st.write(turn["argument"])
    else:
        st.info("Click 'Initiate Multi-Agent Dialectic Debate' to begin.")

st.markdown("### Advanced Collective Intelligence Catalog")
df_adv = pd.DataFrame([
    {"Pattern": k, "Mechanism": v} for k, v in engine.advanced_patterns.items()
])
st.dataframe(df_adv, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does Adversarial Debate eliminate sycophancy and confirmation bias in LLMs?", expanded=False):
    st.markdown("""
    **Answer**:
    A single LLM tends to agree with user premises (**sycophancy**) and defend its own hallucinations.
    In Adversarial Debate, the Opponent agent is explicitly instructed to search for factual errors, ungrounded assumptions, and edge cases in the Proponent's argument. This dialectic tension exposes weaknesses before reaching the judge, resulting in far more balanced outputs.
    """)

with st.expander("Q2: What is the architectural role of subgraphs in Hierarchical Agent Workflows?", expanded=False):
    st.markdown("""
    **Answer**:
    Subgraphs provide **state encapsulation**. A complex child team (e.g. 5-node coding team) has its own state schema, scratchpad buffers, and linting channels. The parent supervisor only needs to know the input and final output. This prevents state schema pollution and allows teams to be developed, tested, and versioned independently.
    """)

with st.expander("Q3: When should an engineer use Borda Count voting over simple Majority Voting?", expanded=False):
    st.markdown("""
    **Answer**:
    Simple majority voting works well when choosing between 2 options. When choosing between 4 or more candidates, majority voting often produces a tie or splits votes among similar options (the spoiler effect).
    **Borda Count** (ranked choice) allows agents to express nuanced second and third preferences, ensuring the winning candidate has the broadest consensus support.
    """)

with st.expander("Q4: How do you prevent multi-agent debate from devolving into an infinite circular argument?", expanded=False):
    st.markdown("""
    **Answer**:
    1. Enforce a strict round limit counter in state (`round_count >= max_rounds`).
    2. The router predicate must terminate the cycle unconditionally when the limit is reached.
    3. The Judge node must be instructed to synthesize a final conclusion rather than asking for more information.
    """)

with st.expander("Q5: What are the cost and latency implications of multi-agent debate in production?", expanded=False):
    st.markdown("""
    **Answer**:
    A 2-round debate between Proponent and Opponent followed by a Judge requires 5 LLM calls (`Pro 1 -> Con 1 -> Pro 2 -> Con 2 -> Judge`).
    This increases token costs by ~5x and latency by ~5x compared to a zero-shot prompt. Debate should be reserved for high-value asynchronous workflows (e.g. strategic planning, legal review, medical analysis) rather than real-time chat widgets.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Debate Cycles from Missing Round Increment")
    st.markdown("""
    *Mechanism*: Forgetting `round_count: state['round_count'] + 1` in the Proponent node update.  
    *Consequence*: `round_count` never reaches `max_rounds`, crashing the server via `GraphRecursionError`.  
    *Fix*: Always increment the iteration counter monotonically in the cycle.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): State Key Collisions in Hierarchical Subgraph Merges")
    st.markdown("""
    *Mechanism*: Child subgraph returns `{"status": "DONE"}` which overwrites the parent graph's `status` channel.  
    *Consequence*: Unintended parent state mutation and corrupted top-level routing.  
    *Fix*: Namespace child outputs explicitly (e.g. `subgraph_research_output`).
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Unbounded Context Windows in Multi-Turn Debate Transcripts")
    st.markdown("""
    *Mechanism*: Accumulating full multi-thousand word speeches across 5 debate rounds into prompt context.  
    *Consequence*: Late-round LLM calls exceed context window limits and token latency explodes.  
    *Fix*: Prompt debate agents to restrict speeches to concise 150-word rebuttals.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Biased Judge Due to Tone and Persuasiveness")
    st.markdown("""
    *Mechanism*: Judge model awards the verdict to the agent using emotional, persuasive rhetoric rather than factual truth.  
    *Consequence*: Eloquent hallucinations win over factual accuracy.  
    *Fix*: Instruct the Judge to evaluate strictly on verifiable citations, logic, and data.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Adversarial Debate**\\nA multi-agent pattern where opposing personas challenge each other before a neutral judge.")
with col_fc2:
    st.info("**Borda Count**\\nA ranked-choice consensus algorithm preventing vote splitting across multiple candidate solutions.")
with col_fc3:
    st.info("**Hierarchical Workflow**\\nA supervisor-worker architecture where specialized teams run as encapsulated subgraphs.")
with col_fc4:
    st.info("**Sycophancy**\\nThe cognitive bias where an LLM blindly agrees with the user's premise instead of evaluating truth.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the collective intelligence benefits of Adversarial Multi-Agent Debate.", value=True)
c2 = st.checkbox("I know how to implement strict round guards to prevent infinite debate cycles.", value=True)
c3 = st.checkbox("I understand how Hierarchical Subgraphs isolate state schemas.", value=True)
c4 = st.checkbox("I can implement Borda Count and Majority Voting consensus mechanisms.", value=True)
c5 = st.checkbox("I know how to design neutral Judge nodes to prevent rhetoric and position bias.", value=True)
c6 = st.checkbox("I understand the cost and latency trade-offs of multi-agent collective intelligence.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 9.3 'Advanced Patterns' with full architectural and practical comprehension!")
