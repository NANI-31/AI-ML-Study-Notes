import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="8.3 Advanced Reasoning Patterns | GenAI Roadmap",
    page_icon="🧬",
    layout="wide",
)

# Custom Styling for GPU-accelerated sidebar & Glassmorphic cards
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🧬 8.3 Advanced Reasoning Patterns — ToT, Self-Consistency & ReAct")
st.markdown(
    """
Welcome to **Lesson 8.3: Advanced Reasoning Patterns**.  
Linear Chain-of-Thought prompting assumes a problem can be solved in a single straight-line deductive chain. 
However, complex enterprise tasks—algorithmic optimization, multi-agent planning, and scientific troubleshooting—require **non-linear exploration, ensemble consensus, and external tool execution**.

Below, we master the **6 Frontier Reasoning Architectures**: Least-to-Most prompting, Self-Consistency majority voting, Tree of Thoughts (ToT), Plan-and-Solve frameworks, Verification prompts, and the ReAct (Reasoning + Acting) loop.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Frontier Reasoning Architectures Matrix
# ==============================================================================
st.header("1. The 6 Frontier Reasoning Architectures Matrix", divider="rainbow")
st.markdown("Comparing non-linear reasoning algorithms across computation, cost, and reliability:")

patterns_matrix_df = pd.DataFrame({
    "Reasoning Architecture": [
        "1. Least-to-Most Prompting",
        "2. Self-Consistency (CoT-SC)",
        "3. Tree of Thoughts (ToT)",
        "4. Plan-and-Solve (PS)",
        "5. Automated Verification",
        "6. ReAct (Reason + Act)"
    ],
    "Core Mechanism": [
        "Decompose into ordered sub-questions; solve sequentially from easiest to hardest",
        "Sample N reasoning chains at Temp > 0; select the majority-vote consensus answer",
        "Explore multiple reasoning branches with heuristic evaluation and backtracking",
        "Explicitly separate the macro planning stage from the micro execution stage",
        "Independent LLM-as-a-Judge checking intermediate constraints against ground truth",
        "Interleave Thought -> Action (Tool Call) -> Observation cycles in a closed loop"
    ],
    "Compute Overhead": [
        "Moderate ($K$ sequential prompt calls)",
        "High ($N$ parallel completions; $N=5$ to $N=10$)",
        "Very High (Search tree with breadth/depth exploration)",
        "Low (Single structured 2-phase prompt)",
        "Moderate (1 additional verification API call)",
        "Variable (Depends on number of tool iterations)"
    ],
    "Best Suited For": [
        "Compositional problems, math word problems with dependencies",
        "Mathematical certainty, critical classification, high-stakes decisions",
        "Complex logic puzzles, strategic game planning, software architecture",
        "Refactoring large legacy codebases, writing technical whitepapers",
        "Compliance enforcement, automated code reviews, medical protocol checks",
        "Autonomous AI agents, live database querying, web browsing agents"
    ]
})
st.dataframe(patterns_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1. Least-to-Most
with st.container(border=True):
    st.subheader("1. Least-to-Most Prompting (Zhou et al., 2022)")
    st.markdown(
        """
Breaks down a composite challenge into a sequence of simpler sub-problems, where the solution to sub-problem 1 is fed as context into sub-problem 2.
"""
    )
    st.code(
        '''prompt = """
QUESTION:
A cloud cluster has 12 worker nodes. Each node runs 4 pods. Each pod handles 25 req/sec.
If traffic surges to 15,000 req/sec, how many additional worker nodes must be provisioned?

LET'S BREAK THIS DOWN INTO SUB-PROBLEMS:
Sub-problem 1: What is the current capacity per worker node?
-> 4 pods * 25 req/sec = 100 req/sec per node.

Sub-problem 2: What is the total current capacity across all 12 nodes?
-> 12 nodes * 100 req/sec = 1,200 req/sec.

Sub-problem 3: How much capacity deficit exists under the 15,000 req/sec surge?
-> 15,000 - 1,200 = 13,800 req/sec deficit.

Sub-problem 4: How many additional nodes are needed to cover 13,800 req/sec?
-> 13,800 / 100 = 138 additional nodes.

FINAL ANSWER: 138 additional nodes must be provisioned.
"""''',
        language="text"
    )

# 2. Self-Consistency
with st.container(border=True):
    st.subheader("2. Self-Consistency (Wang et al., Google Research)")
    st.markdown(
        """
Instead of taking a single greedy response (`temperature=0.0`), generate **5 diverse reasoning paths** with `temperature=0.7`, extract the final answers, and take the **majority-vote consensus**. 
This filters out spurious calculation mistakes and hallucinatory slips.
"""
    )
    st.code(
        '''from collections import Counter
import openai

client = openai.OpenAI()

def solve_with_self_consistency(query: str, n_samples: int = 5) -> str:
    # 1. Sample N diverse reasoning completions
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{query}\\nThink step by step and end with FINAL_ANSWER: [result]"}],
        temperature=0.7,
        n=n_samples
    )
    
    # 2. Extract answer strings from all N completions
    answers = []
    for choice in response.choices:
        content = choice.message.content
        if "FINAL_ANSWER:" in content:
            ans = content.split("FINAL_ANSWER:")[-1].strip().split("\\n")[0]
            answers.append(ans)
            
    # 3. Take majority vote
    consensus_answer, vote_count = Counter(answers).most_common(1)[0]
    return f"Consensus Result: {consensus_answer} ({vote_count}/{n_samples} votes)"''',
        language="python"
    )

# 6. ReAct Loop
with st.container(border=True):
    st.subheader("6. The ReAct (Reasoning + Acting) Framework (Yao et al., 2022)")
    st.markdown(
        """
Synergizes reasoning and action. The model generates a **Thought**, executes an **Action** (calling an external API/tool), observes the **Observation**, and continues until the task is complete.
"""
    )
    st.code(
        '''prompt = """
Answer the user request using the ReAct loop:
Available Tools: [db_query, calculator]

User: What is the average order value for customer ID 9012 in 2026?

Thought 1: I need to fetch all order totals for customer 9012 from the database.
Action 1: db_query(sql="SELECT total_amount FROM orders WHERE customer_id = 9012 AND year = 2026")
Observation 1: [120.0, 450.0, 230.0]

Thought 2: I have 3 orders: $120, $450, and $230. I need to calculate the average.
Action 2: calculator(expr="(120 + 450 + 230) / 3")
Observation 2: 266.67

Thought 3: The calculations are verified. I can now emit the final answer.
Final Answer: The average order value for customer 9012 in 2026 is $266.67.
"""''',
        language="text"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Self-Consistency Workbench
# ==============================================================================
st.header("3. Interactive Workbench: The Self-Consistency Consensus Engine", divider="rainbow")
st.markdown(
    """
Observe how **Self-Consistency (CoT-SC)** achieves statistical robustness over a single greedy prompt:  
**Problem**: *"A distributed Kafka cluster has 3 broker partitions with replication factor 3. If 1 broker crashes, does message ingestion halt?"*
"""
)

st.markdown("#### 5 Sampled Reasoning Paths (Temperature = 0.7):")

paths_data = [
    {"Path": "Sample 1", "Reasoning Trail": "Replication factor 3 means 2 replicas remain active. In-sync replicas (ISR) >= 2 satisfies min.insync.replicas=2.", "Conclusion": "INGESTION CONTINUES"},
    {"Path": "Sample 2", "Reasoning Trail": "Broker 1 down, but leader election triggers automatically. Remaining brokers accept writes.", "Conclusion": "INGESTION CONTINUES"},
    {"Path": "Sample 3 (Flawed)", "Reasoning Trail": "A broker crash always stops cluster ingress until manual restart.", "Conclusion": "INGESTION HALTS"},
    {"Path": "Sample 4", "Reasoning Trail": "With RF=3, losing 1 node leaves quorum intact. Ingestion proceeds uninterrupted.", "Conclusion": "INGESTION CONTINUES"},
    {"Path": "Sample 5", "Reasoning Trail": "Partitions rebalance across remaining 2 online brokers. Traffic continues.", "Conclusion": "INGESTION CONTINUES"}
]

st.dataframe(pd.DataFrame(paths_data), use_container_width=True, hide_index=True)

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    st.success("🎯 **Consensus Result**: `INGESTION CONTINUES` (4 out of 5 votes = 80% confidence).")
with col_sc2:
    st.info("💡 **Error Filtration**: The erroneous conclusion in Sample 3 was automatically out-voted and discarded!")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. Tree of Thoughts & ReAct Architecture", divider="rainbow")

col_ar1, col_ar2 = st.columns([1, 1])

with col_ar1:
    st.markdown("#### Tree of Thoughts (ToT) Exploration Topology")
    dot_tot = """
    digraph TreeOfThoughts {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Root [label="Problem State\\nRoot Node", fillcolor="#334155", fontcolor="#ffffff"];
        B1 [label="Branch 1\\nApproach A", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        B2 [label="Branch 2\\nApproach B", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        B3 [label="Branch 3\\nApproach C", fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];

        Eval1 [label="Heuristic Score: 0.92\\n(Optimal Path)", fillcolor="#047857", fontcolor="#ffffff"];
        Eval2 [label="Heuristic Score: 0.15\\n(Prune / Backtrack)", fillcolor="#991b1b", fontcolor="#ffffff"];

        Root -> B1;
        Root -> B2;
        Root -> B3;
        B1 -> Eval1;
        B2 -> Eval2;
    }
    """
    st.graphviz_chart(dot_tot, use_container_width=False)

with col_ar2:
    st.markdown("#### The ReAct Interleaving Loop")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ 1. THOUGHT: Reason about state and choose action       │
├────────────────────────────────────────────────────────┤
│ 2. ACTION: Emit structured tool call (SQL, Python)     │
├────────────────────────────────────────────────────────┤
│ 3. OBSERVATION: Ingest tool execution payload          │
└───────────────────────────┬────────────────────────────┘
                            │ (Repeat until complete)
                            ▼
              [FINAL DETERMINISTIC ANSWER]""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

ar_tab1, ar_tab2, ar_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with ar_tab1:
    with st.expander("Q1: What is the core mechanical difference between Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT)?", expanded=True):
        st.markdown(
            """
- **Chain-of-Thought (CoT)** is a **linear** sequential path: `Step 0 -> Step 1 -> Step 2 -> ... -> Final Answer`. If the model takes a wrong turn at Step 1, it cannot backtrack and will hallucinate to the end.
- **Tree-of-Thoughts (ToT)** treats reasoning as a **tree search algorithm** (BFS/DFS). It generates multiple candidate thoughts at each step, evaluates their heuristic viability using a self-critique prompt, and actively **backtracks** when a branch hits a dead end. ToT is essential for non-linear problem-solving like theorem proving, code architecture, and multi-constraint scheduling.
"""
        )

    with ar_tab2:
        with st.expander("Trap 1: 'Is Self-Consistency useful if all sampled paths share the exact same hallucinated premise?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, majority voting always finds the truth."*
- **Winning Response**: *"No. Self-Consistency only protects against sampling variance and stochastic calculation errors. If the prompt lacks domain context, or if the model's pre-training weights contain a strong systematic misconception, all $N$ paths will agree on the same wrong answer (correlated error). True accuracy requires external grounding (RAG or tool calling) to anchor the premises before applying consensus voting."*
"""
            )

    with ar_tab3:
        c_ar1, c_ar2, c_ar3 = st.columns(3)
        with c_ar1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Self-Consistency**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Generating multiple reasoning paths at Temperature > 0 and taking majority-vote consensus to eliminate errors.")

        with c_ar2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Tree of Thoughts (ToT)**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Exploring reasoning as a search tree with heuristic evaluation, pruning, and backtracking on dead ends.")

        with c_ar3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**ReAct Framework**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Interleaving Thought (reasoning), Action (tool calling), and Observation (environment feedback).")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 8.3 Summary & Self-Assessment", divider="rainbow")

col_as1, col_as2 = st.columns(2)
with col_as1:
    st.markdown(
        """
### Key Takeaways:
1. **Non-Linear Reasoning**: Move beyond simple linear CoT to ToT and ReAct when tasks require exploration.
2. **Self-Consistency Consensus**: Majority voting across diverse paths filters out stochastic slips.
3. **ReAct Loop**: Interleave thoughts with external tool calls to eliminate math and lookup hallucinations.
4. **Least-to-Most**: Solve dependent sub-problems progressively from simplest to most complex.
"""
    )

with col_as2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain how Self-Consistency majority voting improves reliability over greedy decoding.", value=True, key="chk_8_3_1")
    st.checkbox("I understand the architectural differences between Chain-of-Thought and Tree-of-Thoughts.", value=True, key="chk_8_3_2")
    st.checkbox("I know how to implement the ReAct loop interleaving thoughts, tool actions, and observations.", value=True, key="chk_8_3_3")
    st.checkbox("I can structure a Least-to-Most decomposition prompt for multi-step arithmetic problems.", value=True, key="chk_8_3_4")
