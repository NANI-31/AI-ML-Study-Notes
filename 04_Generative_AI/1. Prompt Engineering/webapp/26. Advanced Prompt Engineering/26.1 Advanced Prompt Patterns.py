import streamlit as st
import json

st.set_page_config(
    page_title="26.1 Advanced Prompt Patterns - Prompt Engineering",
    page_icon="🧠",
    layout="wide"
)

# Header Section
st.title("26.1 Advanced Cognitive Prompt Patterns")
st.caption("Chapter 26: Advanced Prompt Engineering | Domain: 1. Prompt Engineering")
st.markdown("""
When straightforward prompting falls short on complex reasoning, senior prompt engineers employ **advanced cognitive patterns**. 
By orchestrating **Self-Consistency sampling, Self-Critique loops, Reflexion, Multi-Step Decomposition, 
Automated Planning, Semantic Routing, and Iterative Refinement**, models achieve dramatic performance leaps on difficult tasks.
""")

st.divider()

# Section 1: The 8 Advanced Prompt Patterns
st.header("1. Taxonomy of Advanced Cognitive Patterns", divider="rainbow")
st.markdown("""
These eight design patterns transform a standard LLM completion into a multi-stage cognitive engine:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Self-Consistency (Majority Voting)")
    st.markdown("""
    * **Mechanism**: Sample multiple reasoning paths (e.g., $N=5$) at temperature $T=0.7$, then take the **majority vote** on the final answer.
    * **Why it works**: Complex problems have many diverse reasoning paths leading to the correct answer, but erroneous paths diverge randomly.
    """)

    st.subheader("2. Self-Critique & Verification")
    st.markdown("""
    * **Mechanism**: Instruct the model to critically audit its own generated output against a set of validation criteria before finalizing.
    * **Prompt Template**: `"Review your previous answer. Check for: 1. Mathematical consistency, 2. Factuality, 3. Adherence to constraints. Identify flaws and revise."`
    """)

    st.subheader("3. Reflection & Reflexion")
    st.markdown("""
    * **Mechanism**: Store execution feedback (e.g., compiler error trace, test failure) in a short-term episodic buffer, allowing the model to reflect on why it failed and adjust strategy on the next turn.
    """)

    st.subheader("4. Task Decomposition (Least-to-Most)")
    st.markdown("""
    * **Mechanism**: Break a complex problem down into an ordered series of sub-problems, solve the simplest first, and use its output to solve subsequent sub-problems.
    """)

with col2:
    st.subheader("5. Strategic Planning")
    st.markdown("""
    * **Mechanism**: Explicitly mandate creating a step-by-step roadmap *before* generating any code or execution actions.
    * **Benefits**: Prevents the model from writing code into a dead end.
    """)

    st.subheader("6. Semantic Routing")
    st.markdown("""
    * **Mechanism**: Use a fast classifier or embedding similarity to route incoming user queries to specialized prompt pipelines (e.g. Code, Math, Creative, Safety).
    """)

    st.subheader("7. Iterative Refinement")
    st.markdown("""
    * **Mechanism**: Multi-pass drafting: Draft 1 (Core substance) -> Pass 2 (Clarity & brevity) -> Pass 3 (Format compliance).
    """)

    st.subheader("8. Chain-of-Verification (CoVe)")
    st.markdown("""
    * **Mechanism**: Generate baseline response -> Formulate factual verification questions -> Answer verification questions independently -> Revise final response.
    """)

st.divider()

# Section 2: Code Implementations of Cognitive Patterns
st.header("2. Code Patterns: Self-Consistency & Reflexion", divider="rainbow")

pattern_tab = st.selectbox(
    "Choose Pattern to Inspect:",
    ["Pattern A: Self-Consistency Voting", "Pattern B: Reflexion Self-Correction Loop"]
)

if pattern_tab == "Pattern A: Self-Consistency Voting":
    st.markdown("#### Self-Consistency Algorithm in Python")
    st.code('''# Self-Consistency Sampling & Majority Vote
from collections import Counter
import re

def self_consistency_solve(problem_prompt: str, n_samples: int = 5) -> str:
    # 1. Sample N reasoning paths at non-zero temperature
    sampled_answers = []
    
    for _ in range(n_samples):
        # mock_llm_call with temperature=0.7 and CoT instructions
        response = mock_llm_call(
            f"Solve step by step and end with '#### <number>': {problem_prompt}",
            temperature=0.7
        )
        
        # 2. Extract final numerical or categorical answer
        match = re.search(r'####\\s*([^\\n]+)', response)
        if match:
            sampled_answers.append(match.group(1).strip())
            
    # 3. Compute majority vote
    vote_counts = Counter(sampled_answers)
    winning_answer, votes = vote_counts.most_common(1)[0]
    
    return {
        "winning_answer": winning_answer,
        "vote_share": f"{votes}/{n_samples}",
        "all_answers": sampled_answers
    }''', language="python")

else:
    st.markdown("#### Reflexion: Error Feedback Loop")
    st.code('''# Reflexion Pattern with Unit Test Feedback
def reflexion_code_repair(task_spec: str, max_trials: int = 3) -> str:
    current_code = generate_initial_code(task_spec)
    memory_reflections = []
    
    for trial in range(max_trials):
        # Run code against automated test suite
        test_passed, test_error = run_tests(current_code)
        if test_passed:
            return current_code
            
        # Reflection Step: Reflect on test failure
        reflection_prompt = f"""Task: {task_spec}
Current Code:
{current_code}
Test Failure: {test_error}
Reflect on why this code failed the test and describe the exact algorithmic change needed."""
        reflection = mock_llm_call(reflection_prompt)
        memory_reflections.append(reflection)
        
        # Revision Step: Regenerate code with accumulated reflections
        repair_prompt = f"""Task: {task_spec}
Prior Reflections:
{chr(10).join(memory_reflections)}
Write corrected code resolving the failures."""
        current_code = generate_repaired_code(repair_prompt)
        
    return current_code''', language="python")

st.divider()

# Section 3: Interactive Cognitive Pattern Simulator
st.header("3. Interactive Cognitive Pattern Workbench", divider="rainbow")
st.markdown("Observe how advanced patterns resolve complex, trick questions where standard zero-shot prompts fail.")

demo_problem = st.selectbox(
    "Select Benchmark Problem:",
    [
        "Trick Math: The Bat and Ball Problem",
        "Constraint Logic: River Crossing Puzzle",
        "Ambiguous Entity Resolution"
    ]
)

col_z1, col_z2 = st.columns(2)

with col_z1:
    st.markdown("#### Standard Zero-Shot Prompt")
    if demo_problem == "Trick Math: The Bat and Ball Problem":
        st.error("Prompt: 'A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?'")
        st.write("**Model Output**: The ball costs $0.10.")
        st.caption("❌ Failure: Gut-instinct heuristic error ($1.10 - $1.00 = $0.10). If ball is $0.10, bat is $1.10, total is $1.20.")
    elif demo_problem == "Constraint Logic: River Crossing Puzzle":
        st.error("Prompt: 'A farmer has a fox, goose, and bag of grain. How does he cross the river without leaving fox with goose or goose with grain?'")
        st.write("**Model Output**: Takes fox first, then comes back for grain.")
        st.caption("❌ Failure: Leaves goose with grain, causing invalid state violation.")
    else:
        st.error("Prompt: 'Is Paris in France or Texas?'")
        st.write("**Model Output**: Paris is the capital of France.")
        st.caption("❌ Incomplete: Misses that Paris, Texas is also a well-known city.")

with col_z2:
    st.markdown("#### With Cognitive Pattern (Self-Consistency & Verification)")
    if demo_problem == "Trick Math: The Bat and Ball Problem":
        st.success("Reasoning Trace (Chain-of-Verification):")
        st.write("""
        1. Let ball = B. Bat = B + 1.00.
        2. Total: B + (B + 1.00) = 1.10.
        3. 2B = 0.10 -> B = 0.05.
        4. Verification: Bat = $1.05, Ball = $0.05. Total = $1.10.
        **Final Answer**: $0.05 (5 cents).
        """)
    elif demo_problem == "Constraint Logic: River Crossing Puzzle":
        st.success("Reasoning Trace (Plan & Step Decomposition):")
        st.write("""
        1. Step 1: Take goose across. (Fox and grain are safe alone).
        2. Step 2: Return empty. Take fox across.
        3. Step 3: Bring goose back! (Key counter-intuitive move).
        4. Step 4: Leave goose, take grain across.
        5. Step 5: Return empty, take goose across.
        """)
    else:
        st.success("Reasoning Trace (Decomposition & Disambiguation):")
        st.write("Disambiguation: Paris is famously the capital of France, but Paris is also an incorporated city in Lamar County, Texas. Depending on context, both are geographically valid.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is the mathematical and theoretical justification for Self-Consistency sampling?"):
        st.write("""
        Self-consistency leverages the observation that complex reasoning problems require a multi-step sequence of deductions. 
        There are many paths to reach the ground-truth answer, but errors and hallucinations are dispersed randomly across 
        the generation probability space. By marginalizing over diverse reasoning paths via majority voting, 
        the variance of erroneous outputs cancels out while the invariant truth accumulates high consensus.
        """)

    with st.expander("Q: How does Reflexion differ from standard zero-shot self-critique?"):
        st.write("""
        * **Standard Self-Critique**: Single-turn critique where the model evaluates its own prose without external truth signals. 
        Often suffers from self-confirmation bias.
        * **Reflexion**: Grounded in **objective external environment feedback** (compiler errors, unit test exit codes, 
        or environment rewards). It records these feedback traces into persistent verbal memory across multiple iterative trials.
        """)

with qa2:
    with st.expander("Trap to Avoid: Self-Consistency on Low-Entropy Tasks"):
        st.write("""
        Running Self-Consistency (sampling 5x or 10x) on deterministic, simple tasks (like keyword extraction or binary sentiment) 
        multiplies API costs by 5x with zero accuracy benefit. Reserve Self-Consistency exclusively for high-variance reasoning tasks 
        such as mathematical proof generation, complex logic puzzles, or competitive code synthesis.
        """)

    with st.expander("Flashcards: Advanced Patterns"):
        fc_sel = st.selectbox(
            "Select Cognitive Pattern:",
            ["Self-Consistency", "Reflexion", "Chain-of-Verification (CoVe)", "Semantic Routing"]
        )
        if fc_sel == "Self-Consistency":
            st.info("Sampling multiple reasoning paths at T>0 and taking the majority vote on the final answer.")
        elif fc_sel == "Reflexion":
            st.info("An agent pattern that learns from external test failures by maintaining an episodic memory of reflections.")
        elif fc_sel == "Chain-of-Verification (CoVe)":
            st.info("Formulating and answering independent verification questions to audit baseline claims.")
        else:
            st.info("Directing user queries to specialized prompt pipelines based on embedding similarity.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand when to apply Self-Consistency vs single-pass greedy decoding.")
    st.checkbox("I can implement a multi-trial Reflexion error-correction loop in Python.")
with c2:
    st.checkbox("I know how Chain-of-Verification (CoVe) reduces factual hallucinations.")
    st.checkbox("I can design semantic routers to dispatch queries to specialized prompt templates.")
