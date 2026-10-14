import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="8.3 AI Use Cases - LangGraph Curriculum",
    page_icon="💡",
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
# MODULE-SCOPE AI USE CASES ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class AIUseCasesEngine:
    """
    Simulates production AI use cases powered by LangGraph's Send primitive:
    Multi-Document Summarization, Multi-Query Research, Candidate Generation, and Ensemble Consensus.
    """
    def __init__(self):
        self.use_cases = {
            "Multi-Document Processing": "Ingests N documents in parallel and extracts structured entities.",
            "Parallel Document Summarization": "Summarizes sections concurrently and produces an executive brief.",
            "Multi-Query Research": "Breaks complex inquiries into 3 sub-queries and executes search workers.",
            "Parallel Web Research": "Dispatches queries across search APIs and aggregates citations.",
            "Multiple Candidate Generation": "Generates 3 diverse solutions with different temperature/prompts.",
            "Ensemble Reasoning": "Gathers multiple answers and applies majority voting or judge scoring."
        }

    def simulate_ensemble(self, prompt: str, candidate_count: int) -> Dict[str, Any]:
        """
        Simulates generating N candidate solutions via Send and computing consensus.
        """
        candidates = []
        answers_pool = [
            f"Option A: Implement Bulk Synchronous Parallel checkpointer for '{prompt}'",
            f"Option B: Deploy hybrid BM25 + dense vector routing for '{prompt}'",
            f"Option C: Implement Bulk Synchronous Parallel checkpointer for '{prompt}'"
        ]
        
        for i in range(candidate_count):
            ans = answers_pool[i % len(answers_pool)]
            confidence = round(np.random.uniform(0.82, 0.98), 2)
            candidates.append({
                "candidate_id": f"candidate_{i+1}",
                "temperature": 0.2 + (i * 0.3),
                "answer": ans,
                "confidence": confidence
            })

        # Calculate consensus
        from collections import Counter
        ans_counts = Counter(c["answer"] for c in candidates)
        majority_winner, votes = ans_counts.most_common(1)[0]

        return {
            "candidates": candidates,
            "majority_answer": majority_winner,
            "vote_share": f"{votes} / {candidate_count} votes",
            "consensus_ratio": round(votes / candidate_count, 2)
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Applied AI Architectures Powered by the Send API", divider="rainbow")

st.markdown("""
The true power of LangGraph's **Send API** shines in advanced Generative AI architectures.
By combining dynamic worker spawning with barrier reducer aggregation, developers can implement
industry-standard production patterns: **Multi-Query Decomposition**, **Massive Web Research**,
**Best-of-N Candidate Generation**, and **Multi-Agent Ensemble Voting**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    use_cases_blueprint_dot = """
    digraph AIUseCasesArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress & Strategy Selection";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START\\nUser Goal: Solve Complex Task", fillcolor="#1E293B", shape=ellipse];
            Planner [label="Planner / Query Decomposer\\nGenerates dynamic sub-tasks list", fillcolor="#0284C7"];
        }

        subgraph cluster_send_dispatch {
            label="2. Dynamic Parallel Execution (Send API)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Worker_A [label="Research Worker 1\\nExecutes Web Search (Query 1)", fillcolor="#4338CA"];
            Worker_B [label="Research Worker 2\\nExecutes Vector Search (Query 2)", fillcolor="#4338CA"];
            Worker_C [label="Research Worker 3\\nExecutes Database SQL (Query 3)", fillcolor="#4338CA"];
        }

        subgraph cluster_consensus_judge {
            label="3. Consensus & Judge Verification (Reduce)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Reducer_Hub [label="Channel Reducer: Annotated[list, operator.add]\\nCollects all findings & candidates", fillcolor="#0D9488"];
            Judge_Node [label="LLM Judge / Ensemble Voter\\nEvaluates candidate quality & cross-references", fillcolor="#059669"];
            END [label="END\\nVerified Answer Delivered", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Planner;
        Planner -> Worker_A [label="Send('worker_search', {'q': q1})"];
        Planner -> Worker_B [label="Send('worker_search', {'q': q2})"];
        Planner -> Worker_C [label="Send('worker_search', {'q': q3})"];
        Worker_A -> Reducer_Hub;
        Worker_B -> Reducer_Hub;
        Worker_C -> Reducer_Hub;
        Reducer_Hub -> Judge_Node;
        Judge_Node -> END;
    }
    """
    st.graphviz_chart(use_cases_blueprint_dot, use_container_width=True)

st.markdown("""
### Real-World AI Impact: Reducing Hallucinations via Ensemble Diversity
Single LLM completions frequently hallucinate or produce suboptimal reasoning steps.
By spawning 3 diverse candidates via `Send` with varying temperatures or prompt strategies,
an independent LLM judge can cross-examine outputs, select the factual consensus, and reject hallucinated outliers.
""")

# ==============================================================================
# SECTION 2: MULTI-DOCUMENT PROCESSING
# ==============================================================================
st.header("2. Multi-Document Processing: High-Throughput Batch Ingestion Pipelines", divider="rainbow")

st.markdown("""
In enterprise knowledge pipelines, users frequently submit multiple PDFs, CSVs, or web URLs simultaneously.
`Send` maps over the document collection, spawning worker nodes that extract metadata, scrub PII, and generate embeddings concurrently.
""")

code_multidoc = """def batch_ingest_router(state: BatchState) -> list[Send]:
    return [
        Send("process_single_doc", {"doc_id": doc["id"], "file_bytes": doc["bytes"]})
        for doc in state["uploaded_documents"]
    ]
"""
st.code(code_multidoc, language="python")

# ==============================================================================
# SECTION 3: PARALLEL DOCUMENT SUMMARIZATION
# ==============================================================================
st.header("3. Parallel Document Summarization: Recursive Tree and Map-Reduce Summaries", divider="rainbow")

st.markdown("""
For 100-page contracts or 10-K financial reports, sequential processing is prohibitively slow.
Parallel document summarization chunks the report into chapters, maps summarizers to all chapters concurrently,
and merges chapter summaries into a final executive summary.
""")

# ==============================================================================
# SECTION 4: MULTI-QUERY RESEARCH
# ==============================================================================
st.header("4. Multi-Query Research: Parallel Query Decomposition and Sub-Question Solving", divider="rainbow")

st.markdown("""
A complex question like *"Compare the financial performance of NVDA, MSFT, and GOOG in Q3"* is best solved by:
1. Decomposing into 3 sub-queries: `"NVDA Q3 revenue"`, `"MSFT Q3 revenue"`, `"GOOG Q3 revenue"`.
2. Emitting 3 `Send` objects targeting the research worker.
3. Merging the three financial snapshots in the reduce step.
""")

# ==============================================================================
# SECTION 5: PARALLEL WEB RESEARCH
# ==============================================================================
st.header("5. Parallel Web Research: Concurrent Search Execution and Cross-Source Grounding", divider="rainbow")

st.markdown("""
Web searches are I/O bound, taking 500ms–2,000ms per search query.
By dispatching 5 parallel searches via `Send`, total search latency equals **the slowest single search**
rather than the sum of all 5 searches!
""")

# ==============================================================================
# SECTION 6: MULTIPLE CANDIDATE GENERATION
# ==============================================================================
st.header("6. Multiple Candidate Generation: Exploring Diverse Solution Paths", divider="rainbow")

st.markdown("""
Known as **Best-of-N Sampling**, this pattern spawns multiple generator workers with varying temperatures
(e.g. 0.2, 0.7, 1.0) to explore diverse reasoning paths or creative solutions.
""")

code_candidates = """def spawn_candidates_router(state: ReasoningState) -> list[Send]:
    temperatures = [0.2, 0.5, 0.9]
    return [
        Send("reasoning_generator", {"prompt": state["prompt"], "temp": t})
        for t in temperatures
    ]
"""
st.code(code_candidates, language="python")

# ==============================================================================
# SECTION 7: ENSEMBLE REASONING
# ==============================================================================
st.header("7. Ensemble Reasoning: Majority Voting, Weighted Consensus, and Judge Verification", divider="rainbow")

st.markdown("""
Once candidates are generated, the reduce node executes an **Ensemble Arbitrator**:
- **Majority Voting**: Pick the most frequent answer for classification/math problems.
- **LLM-as-a-Judge**: A separate evaluator model ranks candidates against explicit rubrics.
- **Weighted Consensus**: Weights candidates by model size or historical reliability scores.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Query Research and Ensemble Reasoning Pipeline", divider="rainbow")

st.markdown("""
Below is a complete production pipeline showing query decomposition, parallel `Send` worker execution,
and LLM judge consensus selection.
""")

prod_use_cases_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# Step 1: State Contracts
class EnsembleState(TypedDict):
    complex_question: str
    sub_tasks: List[str]
    candidate_solutions: Annotated[List[Dict[str, Any]], operator.add]
    selected_solution: str
    judge_rationale: str

class WorkerState(TypedDict):
    task: str
    strategy: str

# Step 2: Nodes
def planner_node(state: EnsembleState) -> Dict[str, Any]:
    # Decompose into parallel exploration paths
    return {
        "sub_tasks": [
            "Analytical Solution (Mathematical)",
            "Heuristic Solution (Rule-of-Thumb)",
            "Simulation Solution (Monte Carlo)"
        ]
    }

def solver_node(state: WorkerState) -> Dict[str, Any]:
    # Each candidate uses a distinct strategy
    strat = state["strategy"]
    solution = f"Solution via {strat}: Verified optimal routing topology."
    return {
        "candidate_solutions": [{
            "strategy": strat,
            "solution": solution,
            "confidence": 0.92
        }]
    }

def judge_node(state: EnsembleState) -> Dict[str, Any]:
    candidates = state["candidate_solutions"]
    # LLM Judge selects best candidate based on highest confidence
    best = max(candidates, key=lambda c: c["confidence"])
    return {
        "selected_solution": best["solution"],
        "judge_rationale": f"Selected {best['strategy']} with confidence {best['confidence']}."
    }

# Step 3: Send Router
def dispatch_solvers(state: EnsembleState) -> List[Send]:
    return [
        Send("solver", {"task": state["complex_question"], "strategy": t})
        for t in state["sub_tasks"]
    ]

# Step 4: Assemble Graph
def build_ensemble_graph():
    builder = StateGraph(EnsembleState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("solver", solver_node)
    builder.add_node("judge", judge_node)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_solvers)
    builder.add_edge("solver", "judge")
    builder.add_edge("judge", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_ensemble_graph()
    payload = {
        "complex_question": "How do we achieve high-availability in distributed LangGraph deployments?",
        "sub_tasks": [],
        "candidate_solutions": [],
        "selected_solution": "",
        "judge_rationale": ""
    }
    result = app.invoke(payload)
    print("Selected Solution:", result["selected_solution"])
    print("Judge Rationale:", result["judge_rationale"])
'''
st.code(prod_use_cases_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Use-Case Dispatcher and Consensus Evaluator", divider="rainbow")

st.markdown("""
Test Best-of-N candidate generation and ensemble voting interactively.
Enter a query, configure candidate count, and observe how consensus voting eliminates outliers.
""")

engine = AIUseCasesEngine()

col_uw_l, col_uw_r = st.columns([1, 1.2])

with col_uw_l:
    st.subheader("Configure Ensemble Run")
    user_prompt = st.text_input("User Prompt / Problem Statement", value="What is the optimal checkpointer for multi-tenant cloud graphs?")
    num_candidates = st.slider("Number of Candidates to Generate (N)", 2, 5, 3)

    if st.button("🗳️ Generate Candidates & Run Consensus", use_container_width=True):
        st.session_state.ensemble_res = engine.simulate_ensemble(user_prompt, num_candidates)
        st.success("Ensemble consensus evaluated successfully!")

with col_uw_r:
    st.subheader("Consensus & Voting Results")
    if "ensemble_res" in st.session_state:
        res = st.session_state.ensemble_res
        st.metric("Consensus Winner Share", res["vote_share"])
        st.info(f"**Winning Answer**: {res['majority_answer']}")
        
        st.markdown("**Generated Candidate Pool:**")
        df_cands = pd.DataFrame(res["candidates"])
        st.dataframe(df_cands, use_container_width=True)
    else:
        st.info("Click 'Generate Candidates' to evaluate ensemble reasoning.")

st.markdown("### Production AI Use Case Catalog")
df_catalog = pd.DataFrame([
    {"Use Case": k, "Description": v} for k, v in engine.use_cases.items()
])
st.dataframe(df_catalog, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does parallel web research via Send improve agent accuracy compared to single search?", expanded=False):
    st.markdown("""
    **Answer**:
    A single web search is vulnerable to noisy or biased search results from a single search query phrasing.
    By using `Send` to dispatch 3–5 multi-angle sub-queries concurrently:
    1. The agent covers official docs, community forums, and academic papers simultaneously.
    2. Total wall-clock time remains low because all searches run in parallel.
    3. The reduce node cross-references citations, rejecting unsupported claims.
    """)

with st.expander("Q2: What is the Best-of-N pattern in LangGraph and when should it be used?", expanded=False):
    st.markdown("""
    **Answer**:
    Best-of-N generates N candidate responses in parallel (often using higher temperature or diverse reasoning prompts),
    and passes all N answers to an LLM Judge node that scores them against a rubric. It is best used for high-stakes decisions
    (e.g. medical triage, legal drafting, complex math) where accuracy outweighs cost.
    """)

with st.expander("Q3: How do you design an LLM Judge node to avoid bias toward the first candidate in the list?", expanded=False):
    st.markdown("""
    **Answer**:
    LLMs suffer from **position bias** (favoring Candidate 1 or the last candidate).
    Mitigations include:
    1. Shuffling the candidate list randomly before formatting the prompt.
    2. Hiding candidate names and presenting them anonymously (Candidate A, Candidate B).
    3. Asking the judge to score each candidate independently on specific numeric metrics (clarity, accuracy, conciseness) before picking the winner.
    """)

with st.expander("Q4: Can Send workers themselves emit Send objects to create nested multi-level trees?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. A worker node can connect to another conditional edge that returns `List[Send]`.
    This creates a hierarchical tree (e.g. Section -> Paragraph -> Sentence), enabling deep recursive document analysis.
    """)

with st.expander("Q5: How does LangGraph handle streaming when multiple Send workers are running in parallel?", expanded=False):
    st.markdown("""
    **Answer**:
    When using `app.astream(..., stream_mode="updates")`, LangGraph streams individual worker deltas as soon as each worker finishes,
    rather than waiting for the entire superstep to complete. This provides immediate UI feedback to users while other workers are still processing.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Identical Prompts with Zero Temperature in Candidate Generation")
    st.markdown("""
    *Mechanism*: Spawning 5 candidates via `Send` with `temperature=0.0` and the exact same prompt.  
    *Consequence*: All 5 candidates return identical responses, wasting 5x tokens with zero ensemble diversity!  
    *Fix*: Vary temperatures (0.2, 0.7, 1.0) or assign distinct role-play perspectives to each worker.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Non-Determinism in Judge Scoring Causing Flaky Tests")
    st.markdown("""
    *Mechanism*: Using high temperature (`temperature=0.9`) on the LLM Judge node.  
    *Consequence*: The judge produces erratic, inconsistent winners across test runs.  
    *Fix*: Always set `temperature=0.0` and enforce structured Pydantic schemas on judge nodes.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Compounding Latency in Hierarchical Trees")
    st.markdown("""
    *Mechanism*: Nesting 4 levels of Map-Reduce trees where each level takes 3s.  
    *Consequence*: End-to-end latency explodes to 12s+ before user receives initial output.  
    *Fix*: Cap tree depth at 2 tiers (e.g. Document -> Chunk -> Summary).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Token Limit Overflow in the Reduction Node")
    st.markdown("""
    *Mechanism*: Spawning 20 workers that each return 1,000 words, sending 20,000 words into a single reduce node.  
    *Consequence*: Context window saturation and high reduce-stage latency.  
    *Fix*: Require map workers to return concise bullet points (100–200 words each).
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Multi-Query**\\nDecomposing complex queries into multiple sub-questions solved in parallel via `Send`.")
with col_fc2:
    st.info("**Best-of-N**\\nGenerating N diverse candidate answers and selecting the highest-quality candidate via a judge.")
with col_fc3:
    st.info("**LLM-as-a-Judge**\\nUsing a dedicated evaluator model to score candidate solutions against explicit criteria.")
with col_fc4:
    st.info("**Position Bias**\\nThe tendency of evaluator models to favor the first or last candidate in prompt listings.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to implement Multi-Query decomposition using the Send API.", value=True)
c2 = st.checkbox("I understand the Best-of-N candidate generation pattern and temperature variation.", value=True)
c3 = st.checkbox("I can implement an LLM Judge node with structured evaluation rubrics.", value=True)
c4 = st.checkbox("I know how to mitigate position bias in candidate evaluation.", value=True)
c5 = st.checkbox("I understand how streaming modes stream updates from parallel Send workers.", value=True)
c6 = st.checkbox("I can prevent context window overflow during large-scale reduction phases.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 8.3 'AI Use Cases' with full architectural and practical comprehension!")
