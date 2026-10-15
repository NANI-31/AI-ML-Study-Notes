import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="15.3 Reasoning-Oriented Prompting - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionReasoningEngine:
    """
    Simulates production reasoning-oriented prompting architectures:
    - Zero-Shot & Few-Shot Chain-of-Thought (CoT)
    - Structured Decomposition (Least-to-Most)
    - Multi-Path Self-Consistency (Wang et al., 2022) with majority voting
    - Verification and Critique-and-Revise Reflexion loops
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_cot_prompt(self, problem: str) -> str:
        """Constructs a structured Chain-of-Thought scratchpad prompt."""
        return f"""<instruction>
Solve the problem presented in <problem>. 
You must think through this step-by-step before arriving at the final conclusion.
Enclose your reasoning trace inside <think> ... </think> tags.
Provide the final answer inside <answer> ... </answer> tags.
</instruction>

<problem>
{problem.strip()}
</problem>"""

    def generate_decomposition_prompt(self, problem: str) -> str:
        """Constructs a Least-to-Most decomposition prompt."""
        return f"""<instruction>
Break down the complex problem in <problem> into an ordered list of smaller, dependent sub-problems.
Solve each sub-problem sequentially, using the output of earlier steps to solve later steps.
</instruction>

<problem>
{problem.strip()}
</problem>"""

    def simulate_self_consistency(self, problem_type: str, num_paths: int = 5, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Simulates multi-path self-consistency generation.
        Samples multiple stochastic reasoning chains and aggregates the majority consensus.
        """
        if "math" in problem_type.lower():
            true_ans = "42"
            distractor_1 = "38"
            distractor_2 = "44"
        else:
            true_ans = "True"
            distractor_1 = "False"
            distractor_2 = "Inconclusive"

        paths = []
        # Higher temperature adds noise; lower temperature concentrates on true answer
        p_correct = max(0.4, 0.95 - (temperature * 0.35))
        
        for i in range(1, num_paths + 1):
            rand_val = self.rng.rand()
            if rand_val < p_correct:
                ans = true_ans
                trace = f"Path {i}: Evaluated intermediate premise correctly -> derived {ans}."
            elif rand_val < p_correct + (1.0 - p_correct) * 0.6:
                ans = distractor_1
                trace = f"Path {i}: Sign error in step 2 -> derived {ans}."
            else:
                ans = distractor_2
                trace = f"Path {i}: Misinterpreted constraint in step 3 -> derived {ans}."
                
            paths.append({"path_id": i, "reasoning_trace": trace, "extracted_answer": ans})

        # Majority voting
        answers = [p["extracted_answer"] for p in paths]
        counts = pd.Series(answers).value_counts()
        winner = counts.index[0]
        consensus_pct = (counts.iloc[0] / num_paths) * 100

        return {
            "paths": paths,
            "counts": counts.to_dict(),
            "winner": winner,
            "consensus_pct": consensus_pct
        }

    def simulate_critique_and_revise(self, initial_draft: str, rubric: str) -> Dict[str, str]:
        """Simulates a 2-stage critique and revision loop."""
        critique = f"Critique based on rubric [{rubric}]: The response assumes zero network latency and fails to specify exponential backoff on retries."
        revision = f"Revised Solution: Incorporates jittered exponential backoff with max 5 retries to mitigate transient network partitions."
        return {
            "critique": critique,
            "revised_solution": revision
        }

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("15.3 Reasoning-Oriented Prompting")
st.caption("Chapter 15: Prompt Engineering | Domain: 2. Large Language Models")

st.markdown("""
Standard autoregressive language models possess an inherent architectural constraint: 
**they allocate a constant amount of compute per generated token**.
When tasked with a multi-step mathematical proof, logical deduction, or complex algorithmic problem, 
an LLM forced to output the answer immediately must compress all intermediate computations into a single forward pass.

**Reasoning-oriented prompting**—including Chain-of-Thought (CoT), Problem Decomposition (Least-to-Most), 
Self-Consistency sampling, and Critique-and-Revise loops—empowers the model to generate intermediate tokens 
(**"thinking tokens"**) that serve as an external working memory scratchpad.

In this lesson, you will master:
1. The mathematical mechanism of Chain-of-Thought (CoT) and token-allocated computational capacity.
2. Structured reasoning scratchpads (`<think> ... </think>`) and premise verification.
3. Problem decomposition: breaking complex questions into sequential sub-task execution graphs.
4. Self-Consistency: stochastic multi-path sampling and majority-voting answer aggregation.
5. Verification pipelines: automated test generation and factual self-checking.
6. Critique-and-revise patterns: constitutional self-critique and iterative refinement.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Reasoning Topology Spectrum", divider="rainbow")

st.markdown("""
Reasoning strategies progress from linear step-by-step traces to parallel multi-path consensus trees and closed-loop critique graphs.
""")

reasoning_blueprint_dot = """
digraph ReasoningSpectrum {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_cot {
        label = "Topology 1: Linear Chain-of-Thought";
        color = "#334155";
        fontcolor = "#94A3B8";
        PromptLinear [label="Input Query", fillcolor="#0F172A"];
        Step1 [label="Step 1: Premise\\n'2 cans * 3 balls = 6'", fillcolor="#0369A1"];
        Step2 [label="Step 2: Arithmetic\\n'5 original + 6 = 11'", fillcolor="#0369A1"];
        AnsLinear [label="Final Answer: 11", fillcolor="#7C3AED"];
        PromptLinear -> Step1 -> Step2 -> AnsLinear;
    }

    subgraph cluster_sc {
        label = "Topology 2: Multi-Path Self-Consistency";
        color = "#334155";
        fontcolor = "#94A3B8";
        PromptSC [label="Input Query", fillcolor="#0F172A"];
        PathA [label="Path A (T=0.7)\\nYields Ans: 11", fillcolor="#047857"];
        PathB [label="Path B (T=0.7)\\nYields Ans: 11", fillcolor="#047857"];
        PathC [label="Path C (T=0.7)\\nYields Ans: 10", fillcolor="#B45309"];
        MajorityVote [label="Majority Vote\\nWinner: 11 (67%)", fillcolor="#7C3AED"];
        PromptSC -> {PathA PathB PathC} -> MajorityVote;
    }

    subgraph cluster_reflexion {
        label = "Topology 3: Critique and Revise (Reflexion)";
        color = "#334155";
        fontcolor = "#94A3B8";
        Draft [label="Initial Draft", fillcolor="#1E293B"];
        Critique [label="Self-Critique\\n'Assumes zero latency'", fillcolor="#B91C1C"];
        Revision [label="Refined Solution\\n'Adds Jittered Backoff'", fillcolor="#0284C7"];
        Draft -> Critique -> Revision;
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(reasoning_blueprint_dot, use_container_width=True)

st.markdown("""
### The Cognitive Spectrum
1. **Chain-of-Thought (CoT)**: Extends working memory by producing sequential intermediate tokens.
2. **Decomposition (Least-to-Most)**: Breaks complex dependency webs into independent, solvable sub-questions.
3. **Self-Consistency**: Exploits thermodynamic sampling variance to find the most probable consensus path.
4. **Critique-and-Revise**: Introduces reflection loops that identify logical flaws before delivering final output.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - STEP-BY-STEP REASONING
# ==============================================================================
st.header("2. Step-by-Step Reasoning (Chain-of-Thought)", divider="rainbow")

st.markdown("""
Introduced by Jason Wei et al. (Google Brain, 2022), **Chain-of-Thought (CoT) Prompting** demonstrated that 
forcing a model to emit intermediate reasoning steps produces dramatic performance breakthroughs on multi-step reasoning benchmarks 
(GSM8K, MATH, SVAMP).
""")

col_cot1, col_cot2 = st.columns(2)
with col_cot1:
    st.subheader("The Constant Compute Bottleneck")
    st.markdown("""
    * In standard autoregressive generation, a transformer performs the exact same number of matrix multiplications (`O(L * d^2)`) to generate `"42"` as it does to generate `"the"`.
    * If asked: *"Roger has 5 tennis balls. He buys 2 cans of 3 balls. How many does he have? Output only the integer."*
    * The network must compute `5 + (2 * 3) = 11` within a single forward pass layer traversal!
    * Small errors in hidden layer activations lead to immediate wrong guesses.
    """)

with col_cot2:
    st.subheader("The Scratchpad Mechanism")
    st.markdown("""
    * **Zero-Shot CoT (Kojima et al., 2022)**: Simply appending the trigger phrase:  
      *`"Let's think step by step."`*  
      causes the model to generate intermediate steps: *"First, 2 cans of 3 balls is 6. Then 5 + 6 is 11."*
    * **Autoregressive Feedback**: Because each generated token is added to the KV Cache, **subsequent tokens attend directly to the computed intermediate results**! 
    * The model uses the text buffer as an **external working memory register**.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - STRUCTURED REASONING
# ==============================================================================
st.header("3. Structured Reasoning (Scratchpad Architectures)", divider="rainbow")

st.markdown("""
In enterprise software, unstructured free-form reasoning is difficult to parse. 
Production applications demand **Structured Scratchpads** that clearly segregate intermediate thinking from the final result.
""")

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    st.markdown("#### The `<think>` Tag Pattern (DeepSeek-R1 / OpenAI o1)")
    st.markdown("""
    Modern reasoning foundation models explicitly structure their generation into two distinct token blocks:
    ```text
    <think>
    1. Identify user requirements: calculate compound interest over 5 years.
    2. Formula: A = P * (1 + r/n)^(nt)
    3. P = 10000, r = 0.05, n = 1, t = 5
    4. Compute 1.05^5 = 1.27628
    5. A = 10000 * 1.27628 = 12762.8
    </think>
    Final Answer: $12,762.80
    ```
    """)

with col_sc2:
    st.markdown("#### Production Advantages of Structured Scratchpads")
    st.markdown("""
    * **Clean UI Streaming**: Downstream web apps can collapse or hide the `<think>` block in an expandable accordion while displaying the clean final answer to the user.
    * **Auditability & Logging**: The reasoning trace can be stored in telemetry databases to debug why a model recommended a specific decision.
    * **Format Safety**: Downstream parsers can extract the contents of `<answer>` directly without complex regex heuristics.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - DECOMPOSITION
# ==============================================================================
st.header("4. Decomposition (Least-to-Most Prompting)", divider="rainbow")

st.markdown("""
Introduced by Denny Zhou et al. (Google Research, 2022), **Least-to-Most Prompting** solves complex tasks by 
explicitly decomposing a problem into an ordered DAG (Directed Acyclic Graph) of simpler sub-problems.
""")

st.markdown("""
#### The Two-Phase Decomposition Pipeline
1. **Phase 1: Decomposition**: The model breaks the primary query into sub-questions:
   ```text
   Original: "How many liters of fuel does an Airbus A380 consume flying from London to NYC with a 20kt headwind?"
   Sub-questions:
   1. What is the flight distance between London and NYC? (~5,570 km)
   2. What is the average cruise speed with a 20kt headwind? (~850 km/h)
   3. What is the total flight time? (~6.55 hours)
   4. What is the hourly fuel burn rate of an A380? (~11,000 kg/h)
   5. Calculate total liters using fuel density (0.8 kg/L).
   ```
2. **Phase 2: Sequential Resolution**: Solve Sub-question 1 -> inject answer into prompt for Sub-question 2 -> iterate until the final answer is computed.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - SELF-CONSISTENCY
# ==============================================================================
st.header("5. Self-Consistency", divider="rainbow")

st.markdown("""
Introduced by Xuezhi Wang et al. (Google Research, 2022), **Self-Consistency** replaces greedy decoding with 
**stochastic multi-path sampling and majority voting**.
""")

col_scy1, col_scy2 = st.columns(2)
with col_scy1:
    st.subheader("The Self-Consistency Algorithm")
    st.markdown("""
    1. Prompt the model with Chain-of-Thought instructions.
    2. Set temperature `T = 0.5 to 0.7` to introduce sampling diversity.
    3. Generate `N` independent reasoning paths (typically `N = 5, 10, or 20`).
    4. Parse the final answers from all `N` paths.
    5. Select the **marginal majority answer**:
       ```text
       Answer_Final = argmax_a Sum_{i=1}^N 1(Answer(Path_i) == a)
       ```
    """)

with col_scy2:
    st.subheader("Why Self-Consistency Outperforms Greedy Search")
    st.markdown("""
    * **Greedy Search Fragility**: A single greedy path can make an early arithmetic slip, locking the model into an incorrect branch.
    * **Path Diversity**: Multiple diverse paths can approach the problem from different angles (algebraic vs graphical vs iterative).
    * **Consensus Convergence**: Incorrect reasoning paths fail in many different random ways, while correct reasoning paths converge on the exact same true answer!
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - VERIFICATION
# ==============================================================================
st.header("6. Verification (Chain-of-Verification / CoVe)", divider="rainbow")

st.markdown("""
Introduced by Dhruv Madeka et al. (Meta AI, 2023), **Chain-of-Verification (CoVe)** is an architectural pattern 
designed to systematically detect and eradicate factual hallucinations in model completions.
""")

st.markdown("""
| Stage | Objective | Execution Prompt |
| :--- | :--- | :--- |
| **1. Baseline Generation** | Generate initial candidate answer | *"Draft an initial response to the user query."* |
| **2. Verification Planning** | Generate fact-checking questions | *"List 3 to 5 independent factual questions to verify the claims made in the baseline draft."* |
| **3. Independent Execution** | Answer verification questions without seeing draft | *"Answer each verification question objectively without bias from the draft."* |
| **4. Final Synthesis** | Revise draft using verified facts | *"Cross-reference the original draft with the verified answers. Output the corrected response."* |
""")

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - CRITIQUE-AND-REVISE PATTERNS
# ==============================================================================
st.header("7. Critique-and-Revise Patterns (Reflexion)", divider="rainbow")

st.markdown("""
Popularized by the **Reflexion framework (Shinn et al., 2023)** and Anthropic's **Constitutional AI**, 
critique-and-revise patterns introduce an explicit self-evaluation loop before finalizing generation.
""")

col_cr1, col_cr2 = st.columns(2)
with col_cr1:
    st.markdown("#### The 3-Step Iterative Loop")
    st.markdown("""
    1. **Actor Generates Draft**: Produces code, analysis, or draft text.
    2. **Evaluator Critiques**: Evaluates draft against an explicit rubric (e.g. *"Check for: SQL injection vulnerabilities, missing pagination, unindexed JOINs"*).
    3. **Actor Refines**: Re-generates the solution incorporating the concrete feedback from the critique.
    """)

with col_cr2:
    st.markdown("#### Stopping Criteria")
    st.markdown("""
    * **Maximum Iteration Bound**: Limit critique loops to `max_iterations = 2 or 3` to prevent infinite loops.
    * **Deterministic Unit Tests**: In coding pipelines, stop refining as soon as execution tests pass cleanly.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Path Self-Consistency & Reflexion Engine in Python", divider="rainbow")

st.markdown("""
The following self-contained class demonstrates automated Chain-of-Thought generation, 
stochastic multi-path self-consistency with majority voting, and critique-revision loops.
""")

st.code("""
import numpy as np
import pandas as pd
from typing import Dict, Any

class ProductionReasoningEngine:
    \"\"\"
    Simulates production reasoning-oriented prompting architectures.
    \"\"\"
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_cot_prompt(self, problem: str) -> str:
        return f\"<instruction>\\nSolve step-by-step. Put reasoning inside <think> and final answer inside <answer>.\\n</instruction>\\n\\n<problem>\\n{problem.strip()}\\n</problem>\"

    def simulate_self_consistency(self, num_paths: int = 5, temperature: float = 0.7) -> Dict[str, Any]:
        paths = []
        p_correct = max(0.4, 0.95 - (temperature * 0.35))
        for i in range(1, num_paths + 1):
            ans = "42" if self.rng.rand() < p_correct else "38"
            paths.append({"path_id": i, "extracted_answer": ans})
        
        counts = pd.Series([p["extracted_answer"] for p in paths]).value_counts()
        return {
            "winner": counts.index[0],
            "consensus_pct": (counts.iloc[0] / num_paths) * 100,
            "counts": counts.to_dict()
        }
""", language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Reasoning Strategy Simulator & Majority Voting Console", divider="rainbow")

engine = ProductionReasoningEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("1. Reasoning Simulation Configuration")
    user_prob = st.text_area(
        "Complex Logic / Math Problem",
        value="A farmer has 15 sheep. All but 8 die in a storm. He then buys twice as many as he has left. How many sheep does he have now?"
    )
    user_paths = st.slider("Self-Consistency Sample Paths (N)", min_value=3, max_value=21, value=7, step=2)
    user_temp = st.slider("Sampling Temperature", min_value=0.1, max_value=1.2, value=0.7, step=0.1)

    sc_result = engine.simulate_self_consistency("math", num_paths=user_paths, temperature=user_temp)

with wb_col2:
    st.subheader("2. Multi-Path Majority Voting Consensus")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.metric("Consensus Winner Answer", f"Answer: {sc_result['winner']}")
    with col_w2:
        st.metric("Consensus Confidence", f"{sc_result['consensus_pct']:.1f}%", f"{user_paths} paths")

    st.markdown("##### Vote Distribution Across Sampled Paths:")
    vote_df = pd.DataFrame(list(sc_result["counts"].items()), columns=["Candidate Answer", "Vote Count"])
    st.bar_chart(vote_df.set_index("Candidate Answer"))

    with st.expander("Inspect All Sampled Reasoning Paths"):
        for p in sc_result["paths"]:
            st.write(f"**Path {p['path_id']}**: {p['reasoning_trace']}")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: Why does Chain-of-Thought (CoT) prompting improve multi-step reasoning accuracy?"):
        st.write("""
        In autoregressive transformers, computation per token is constant. 
        When forced to answer immediately without thinking tokens, the model must solve the entire problem in a single feed-forward pass.
        CoT allows the model to generate intermediate steps into its output stream. 
        These generated tokens enter the KV Cache context, allowing subsequent attention heads to attend directly to verified sub-computations.
        """)
    with st.expander("Q2: How does Self-Consistency differ from Beam Search decoding?"):
        st.write("""
        Beam Search seeks to maximize the single sequence probability `P(y | x)` by maintaining the top-k highest probability tokens at each step. It often leads to repetitive or degenerate local optima.
        Self-Consistency samples diverse, independent reasoning chains using non-zero temperature (`T=0.7`), extracts the final answers from each path, and applies marginal majority voting. 
        It prioritizes consensus of the final answer rather than likelihood of individual word sequences.
        """)
    with st.expander("Q3: What is the primary limitation of Chain-of-Thought prompting?"):
        st.write("""
        CoT significantly increases generation token counts, multiplying inference latency and API cost.
        Additionally, CoT does not help on simple factual lookup tasks where reasoning is unnecessary (e.g. *"What is the capital of France?"*), and can occasionally cause the model to rationalize incorrect initial assumptions.
        """)

with q2:
    with st.expander("Q4: What is Least-to-Most Prompting and when should you use it?"):
        st.write("""
        Least-to-Most prompting decomposes a problem into an ordered list of dependent sub-problems, solving them sequentially.
        It is ideal for problems with strict sequential dependencies (e.g., multi-step word problems, symbolic manipulation, long-horizon planning) where solving step 1 is a prerequisite for understanding step 2.
        """)
    with st.expander("Q5: How does the Reflexion framework implement self-correction without weight updates?"):
        st.write("""
        Reflexion (Shinn et al.) maintains a short-term memory buffer of previous failed attempts, external environment feedback (or unit test errors), and self-generated critiques.
        On subsequent attempts, the agent conditions its prompt on these past critique notes, steering away from previously identified errors entirely within the context window.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Forcing Single-Token Output on Complex Reasoning**
    * **Failure**: Writing *"Solve this calculus integral. Output only the final number."*
    * **Impact**: Eliminates the working memory scratchpad; model error rates exceed 70%.
    * **Remedy**: Always allow intermediate thinking tokens, either in `<think>` tags or step-by-step prose.
    """)
    st.error("""
    **🚨 Production Trap 2: Running Self-Consistency at Temperature Zero**
    * **Failure**: Generating 10 paths for majority voting while leaving `temperature = 0.0`.
    * **Impact**: Greedy decoding produces 10 identical paths, completely defeating the purpose of multi-path consensus.
    * **Remedy**: Set `temperature = 0.5 to 0.7` to introduce diverse reasoning trajectories.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Unbounded Critique-and-Revise Loops**
    * **Failure**: Letting an agent critique and refine its output until "perfection" without an iteration ceiling.
    * **Impact**: The model oscillates endlessly between slight stylistic variations, burning budget and exceeding timeout SLAs.
    * **Remedy**: Enforce a strict `max_refinements = 2` guardrail.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Sycophantic Self-Verification**
    * **Failure**: Asking the model: *"Did you make any mistakes in the answer above?"*
    * **Impact**: The model will almost always justify its prior completion rather than identifying subtle errors.
    * **Remedy**: Use independent Chain-of-Verification (CoVe) questions executed in an isolated prompt without seeing the original draft.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Zero-Shot CoT Trigger Phrase"):
        st.write("What 5-word phrase triggers Zero-Shot Chain-of-Thought in foundation models?")
        st.info("'Let's think step by step.' (Kojima et al., 2022)")
    with st.expander("⚡ Flashcard 2: Self-Consistency Decoding Mode"):
        st.write("Does Self-Consistency use greedy search or temperature sampling?")
        st.info("Temperature sampling (typically T = 0.5 - 0.7) to generate diverse paths.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: Structured Scratchpad Benefit"):
        st.write("Why are `<think>` tags preferred over raw CoT in production apps?")
        st.info("Allows clean UI separation, audit logging, and deterministic `<answer>` parsing.")
    with st.expander("⚡ Flashcard 4: CoVe Core Principle"):
        st.write("How does Chain-of-Verification prevent confirmation bias?")
        st.info("By executing verification questions independently without passing the original draft.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of Reasoning-Oriented Prompting before advancing to Lesson 15.4 (Structured Outputs):
""")

chk1 = st.checkbox("I understand why intermediate tokens act as working memory in autoregressive models.", value=True)
chk2 = st.checkbox("I can construct structured scratchpads using <think> and <answer> boundary tags.", value=True)
chk3 = st.checkbox("I know how Least-to-Most prompting decomposes complex queries into sequential DAGs.", value=True)
chk4 = st.checkbox("I can implement Self-Consistency with stochastic sampling and majority voting.", value=True)
chk5 = st.checkbox("I understand the 4 stages of Chain-of-Verification (CoVe) to stop hallucination.", value=True)
chk6 = st.checkbox("I know how to bound Reflexion critique-and-revise loops to prevent execution runaway.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 15.4: Structured Outputs.")
