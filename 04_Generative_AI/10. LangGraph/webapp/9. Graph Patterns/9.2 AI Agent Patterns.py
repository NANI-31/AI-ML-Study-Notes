import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="9.2 AI Agent Patterns - LangGraph Curriculum",
    page_icon="🤖",
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
# MODULE-SCOPE AGENT PATTERN DISPATCHER FOR WORKBENCH & PIPELINE
# ==============================================================================
class AgentPatternDispatcher:
    """
    Simulates the 8 primary AI Agent architectural patterns:
    Prompt->LLM, ReAct (LLM->Tool->LLM), Router->Specialist, Planner->Executor,
    Generator->Evaluator, Generator->Critic->Reviser, Research->Synthesis, Retrieve->Reason->Answer.
    """
    def __init__(self):
        self.catalog = {
            "Prompt -> LLM -> Response": "Zero-shot direct generation without tools or memory.",
            "LLM -> Tool -> LLM": "ReAct loop with tool call execution and observation reflection.",
            "Router -> Specialist": "Intent classification directing to fine-tuned or domain expert.",
            "Planner -> Executor": "Decomposes multi-step goals into a task list and executes sequentially.",
            "Generator -> Evaluator": "Two-node pattern where an independent evaluator scores drafts.",
            "Generator -> Critic -> Reviser": "Three-node loop incorporating explicit critique notes into revisions.",
            "Research -> Synthesis": "Multi-query parallel search aggregation and briefing.",
            "Retrieve -> Reason -> Answer": "Agentic RAG with query reformulation and context grounding."
        }

    def simulate_pattern(self, pattern_name: str, goal: str) -> List[Dict[str, Any]]:
        """
        Executes a simulated multi-step trajectory for the selected agent pattern.
        """
        steps = []
        if pattern_name == "Planner -> Executor":
            steps = [
                {"role": "Planner Node", "action": f"Decomposed '{goal}' into: 1. Fetch data, 2. Run analysis, 3. Generate chart."},
                {"role": "Executor Node (Step 1)", "action": "Fetched 500 records from analytics database."},
                {"role": "Executor Node (Step 2)", "action": "Computed linear regression metrics and p-values."},
                {"role": "Executor Node (Step 3)", "action": "Generated output visualization and final response."}
            ]
        elif pattern_name == "LLM -> Tool -> LLM":
            steps = [
                {"role": "LLM Node", "action": f"Reasoned: Requires weather API to answer '{goal}' -> Emitted tool_call 'get_weather'."},
                {"role": "Tool Node", "action": "Executed 'get_weather(city='San Francisco')' -> Result: 62 F, Sunny."},
                {"role": "LLM Node", "action": "Synthesized observation into natural language answer."}
            ]
        elif pattern_name == "Generator -> Critic -> Reviser":
            steps = [
                {"role": "Generator Node", "action": f"Drafted initial version for: '{goal}'."},
                {"role": "Critic Node", "action": "Identified shortcomings: Missing unit tests and exception handling."},
                {"role": "Reviser Node", "action": "Revised code incorporating critic feedback -> Produced production-ready draft."}
            ]
        else: # Generic pattern trace
            steps = [
                {"role": "Ingress Node", "action": f"Received goal: '{goal}'."},
                {"role": "Core Reasoner Node", "action": f"Applied pattern '{pattern_name}'."},
                {"role": "Terminal Node", "action": "Completed goal with verified assertions."}
            ]
        return steps


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Complete Taxonomy of Generative AI Agent Design Patterns", divider="rainbow")

st.markdown("""
Over the past two years of Generative AI engineering, eight distinct architectural agent patterns have emerged.
These patterns govern how language models interact with tools, external memory, rubrics, and each other.
LangGraph provides the first unified runtime capable of expressing **all eight patterns natively**
using its core state machine primitives (`StateGraph`, `Send`, and `Command`).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    agent_patterns_blueprint_dot = """
    digraph AgentPatternsTaxonomy {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_react {
            label="Pattern A: ReAct (LLM -> Tool -> LLM)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            A_LLM [label="LLM (Reason)", fillcolor="#4338CA"];
            A_Tool [label="ToolNode (Act)", fillcolor="#0D9488"];
            A_LLM -> A_Tool [label="tool_calls"];
            A_Tool -> A_LLM [label="ToolMessage"];
        }

        subgraph cluster_plan_solve {
            label="Pattern B: Planner -> Executor";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            B_Plan [label="Planner (Decompose)", fillcolor="#0284C7"];
            B_Exec [label="Executor (Step N)", fillcolor="#334155"];
            B_Plan -> B_Exec [label="task_list"];
            B_Exec -> B_Exec [label="Loop until done"];
        }

        subgraph cluster_eval_opt {
            label="Pattern C: Generator -> Critic -> Reviser";
            style="filled";
            color="#1E293B";
            fontcolor="#EAB308";
            C_Gen [label="Generator", fillcolor="#334155"];
            C_Critic [label="Critic (Rubric)", fillcolor="#854D0E"];
            C_Rev [label="Reviser", fillcolor="#0D9488"];
            C_Gen -> C_Critic -> C_Rev;
            C_Rev -> C_Critic [label="Iterate"];
        }

        A_Tool -> B_Plan [style=invis];
    }
    """
    st.graphviz_chart(agent_patterns_blueprint_dot, use_container_width=True)

st.markdown("""
### The Cognitive Spectrum: Zero-Shot to Autonomous Multi-Agent
As tasks grow in ambiguity and complexity, systems transition from simple prompt-response chains
to ReAct loops, then to plan-and-solve hierarchies, and finally to collaborative multi-agent teams.
""")

# ==============================================================================
# SECTION 2: PROMPT -> LLM -> RESPONSE
# ==============================================================================
st.header("2. Prompt -> LLM -> Response: Direct Zero-Shot Inference Pipeline", divider="rainbow")

st.markdown("""
The foundational baseline pattern. An incoming user prompt is packaged into a system template,
passed to an LLM, and the response is returned directly to the user.
Best suited for straightforward copy generation, summarization of small texts, and direct question answering.
""")

code_p1 = """# Zero-shot baseline
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
"""
st.code(code_p1, language="python")

# ==============================================================================
# SECTION 3: LLM -> TOOL -> LLM
# ==============================================================================
st.header("3. LLM -> Tool -> LLM: Tool-Augmented Reasoning and ReAct Loops", divider="rainbow")

st.markdown("""
The **ReAct (Reason + Act)** pattern equips the LLM with an environment.
The model generates `tool_calls`, a `ToolNode` executes them, and the resulting observations
are fed back to the model to generate the final synthesized response or trigger additional tools.
""")

code_p2 = """from langgraph.prebuilt import ToolNode, tools_condition

builder.add_node("agent", call_model_with_tools)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent") # ReAct Cycle!
"""
st.code(code_p2, language="python")

# ==============================================================================
# SECTION 4: ROUTER -> SPECIALIST
# ==============================================================================
st.header("4. Router -> Specialist: Multi-Domain Triage and Semantic Dispatch", divider="rainbow")

st.markdown("""
In the **Router-Specialist Pattern**, a front-line triage node classifies user intent
and routes execution to a specialized domain agent (e.g. Code Agent, Legal Agent, Support Agent).
This keeps system prompts focused and prevents domain interference.
""")

# ==============================================================================
# SECTION 5: PLANNER -> EXECUTOR
# ==============================================================================
st.header("5. Planner -> Executor: Plan-and-Solve Hierarchical Orchestration", divider="rainbow")

st.markdown("""
In the **Plan-and-Solve Pattern**, execution is decoupled into two phases:
1. **Planner**: Creates an explicit, step-by-step numbered plan.
2. **Executor**: Steps through each item in the plan, updating a `completed_steps` state list.
This dramatically outperforms zero-shot reasoning on multi-step mathematical and research tasks.
""")

# ==============================================================================
# SECTION 6: GENERATOR -> EVALUATOR
# ==============================================================================
st.header("6. Generator -> Evaluator: Dual-Model Verification and Quality Gates", divider="rainbow")

st.markdown("""
A two-node pattern where Node A produces a draft and Node B evaluates it against a numeric scoring rubric.
If the score fails, the evaluator routes back to the generator with error hints.
""")

# ==============================================================================
# SECTION 7: GENERATOR -> CRITIC -> REVISER
# ==============================================================================
st.header("7. Generator -> Critic -> Reviser: Three-Node Iterative Self-Correction", divider="rainbow")

st.markdown("""
An extension of the Generator-Evaluator pattern featuring a dedicated **Reviser node**:
- **Generator**: Drafts initial version.
- **Critic**: Provides qualitative, point-by-point critique feedback.
- **Reviser**: Takes *both* the original draft and the critique notes to synthesize an improved revision.
""")

# ==============================================================================
# SECTION 8: RESEARCH -> SYNTHESIS
# ==============================================================================
st.header("8. Research -> Synthesis: Parallel Knowledge Ingestion and Deep Briefing", divider="rainbow")

st.markdown("""
A two-stage Map-Reduce pattern where multiple research questions are investigated concurrently via the `Send` API,
and an aggregator node synthesizes findings into a unified intelligence brief.
""")

# ==============================================================================
# SECTION 9: RETRIEVE -> REASON -> ANSWER
# ==============================================================================
st.header("9. Retrieve -> Reason -> Answer: Advanced Agentic RAG Workflows", divider="rainbow")

st.markdown("""
Unlike naive RAG (which blindly injects vector search results into prompts),
**Agentic RAG** reformulates queries, grades document relevance, queries multiple sources,
and falls back to web search if retrieved documents lack sufficient answers.
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Plan-and-Execute Agent Pattern in LangGraph", divider="rainbow")

st.markdown("""
Below is a complete, runnable script implementing the **Plan-and-Execute Pattern** in LangGraph.
A planner node constructs a plan, an executor executes tasks iteratively, and a synthesizer formats the final delivery.
""")

prod_plan_exec_code = '''from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: Define Plan-and-Solve State
class PlanSolveState(TypedDict):
    input_goal: str
    plan: List[str]
    past_steps: List[str]
    current_step_index: int
    final_output: Optional[str]

# Nodes
def planner_node(state: PlanSolveState) -> Dict[str, Any]:
    goal = state["input_goal"]
    # Decompose into 3 actionable steps
    constructed_plan = [
        f"Step 1: Gather API telemetry for '{goal}'",
        f"Step 2: Calculate 99th percentile response latency",
        f"Step 3: Draft executive performance report"
    ]
    return {
        "plan": constructed_plan,
        "current_step_index": 0,
        "past_steps": []
    }

def executor_node(state: PlanSolveState) -> Dict[str, Any]:
    idx = state["current_step_index"]
    active_task = state["plan"][idx]
    
    # Simulate execution of active task
    executed_log = f"Completed: {active_task} with status 200 OK"
    
    return {
        "current_step_index": idx + 1,
        "past_steps": state["past_steps"] + [executed_log]
    }

def synthesizer_node(state: PlanSolveState) -> Dict[str, Any]:
    steps = state["past_steps"]
    summary = f"Successfully executed {len(steps)} plan steps to achieve goal: '{state['input_goal']}'"
    return {"final_output": summary}

# Router Predicate
def plan_progress_router(state: PlanSolveState) -> str:
    if state["current_step_index"] < len(state["plan"]):
        return "continue_executing"
    return "synthesize"

# Step 2: Build Graph Topology
def build_plan_and_solve_graph():
    builder = StateGraph(PlanSolveState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    
    builder.add_conditional_edges(
        "executor",
        plan_progress_router,
        {
            "continue_executing": "executor", # Cycle back until plan finished!
            "synthesize": "synthesizer"
        }
    )
    builder.add_edge("synthesizer", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_plan_and_solve_graph()
    job = {
        "input_goal": "Analyze Q3 Cloud Infrastructure Telemetry",
        "plan": [],
        "past_steps": [],
        "current_step_index": 0,
        "final_output": None
    }
    result = app.invoke(job)
    print("Final Output:", result["final_output"])
    print("Past Steps Log:", result["past_steps"])
'''
st.code(prod_plan_exec_code, language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Agent Pattern Dispatcher and Execution Trace Visualizer", divider="rainbow")

st.markdown("""
Compare how different agent patterns approach the same user goal.
Select an agent pattern, enter a target goal, and review its simulated execution trajectory.
""")

dispatcher = AgentPatternDispatcher()

col_aw_l, col_aw_r = st.columns([1, 1.2])

with col_aw_l:
    st.subheader("Configure Agent Pattern Run")
    selected_pattern = st.selectbox("Select AI Agent Pattern", list(dispatcher.catalog.keys()))
    st.caption(f"**Pattern Description**: {dispatcher.catalog[selected_pattern]}")

    test_goal = st.text_input("Agent Goal / Task", value="Debug latency spikes in distributed database cluster")

    if st.button("🤖 Simulate Agent Execution Trajectory", use_container_width=True):
        st.session_state.agent_trace = dispatcher.simulate_pattern(selected_pattern, test_goal)
        st.success(f"Executed trajectory for '{selected_pattern}'!")

with col_aw_r:
    st.subheader("Agent Execution Trajectory")
    if "agent_trace" in st.session_state:
        for idx, step in enumerate(st.session_state.agent_trace, 1):
            st.markdown(f"**Step {idx}: `{step['role']}`**")
            st.info(step["action"])
    else:
        st.info("Click 'Simulate Agent Execution Trajectory' to view step-by-step actions.")

st.markdown("### The 8 AI Agent Patterns Reference Matrix")
df_catalog = pd.DataFrame([
    {"Pattern Name": k, "Core Architecture": v} for k, v in dispatcher.catalog.items()
])
st.dataframe(df_catalog, use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What are the primary trade-offs between ReAct (LLM->Tool->LLM) and Plan-and-Solve?", expanded=False):
    st.markdown("""
    **Answer**:
    - **ReAct**: Selects one tool at a time, evaluating the result before deciding the next step. Excellent for exploratory or unpredictable tasks, but prone to getting side-tracked or looping on complex multi-step tasks.
    - **Plan-and-Solve**: Plans the entire sequence upfront before executing. Excellent for complex multi-step reasoning with predictable sub-goals, reducing token costs by avoiding repetitive tool deliberation on every turn.
    """)

with st.expander("Q2: Why does the Generator-Critic-Reviser pattern outperform a simple Generator-Evaluator pattern?", expanded=False):
    st.markdown("""
    **Answer**:
    In Generator-Evaluator, the evaluator typically emits only a boolean or numeric score (`Pass / Fail` or `Score: 0.6`).
    The generator must guess what went wrong. In Generator-Critic-Reviser, the **Critic node** provides explicit natural language actionable critique notes (e.g. *"Missing null-check on line 14"*), giving the Reviser exact instructions on how to patch the draft.
    """)

with st.expander("Q3: How does Agentic RAG differ from traditional Naive RAG?", expanded=False):
    st.markdown("""
    **Answer**:
    Naive RAG executes a fixed sequence: `Query -> Embed -> VectorDB -> Prompt -> Answer`.
    Agentic RAG wraps retrieval in a dynamic decision graph:
    1. Evaluates if the query needs rewriting.
    2. Evaluates if retrieved chunks are actually relevant.
    3. Triggers alternative search APIs if relevance score is low.
    4. Evaluates if the generated answer is grounded in context before returning it.
    """)

with st.expander("Q4: What is the Router-Specialist pattern and how does it prevent prompt bloat?", expanded=False):
    st.markdown("""
    **Answer**:
    In a monolithic agent, the system prompt must contain rules and few-shot examples for dozens of diverse tasks (coding, billing, legal, support), causing prompt bloat and confusing the model.
    In the Router-Specialist pattern, a fast router classifies the user intent and delegates to a specialized agent whose prompt contains *only* the guidelines relevant to that specific domain.
    """)

with st.expander("Q5: How do you prevent Plan-and-Solve agents from continuing to execute when a step fails catastrophically?", expanded=False):
    st.markdown("""
    **Answer**:
    In the executor node, wrap task execution in validation gates. If a step returns a fatal error:
    1. Mark `has_error = True` in state.
    2. Direct the router to a **Re-Planner node** that modifies the remaining plan to find an alternative workaround rather than mindlessly executing doomed subsequent steps.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite ReAct Loops from Unmatched Tool Names")
    st.markdown("""
    *Mechanism*: LLM calls a tool `calc_revenue()`, but the registered tool is named `calculate_revenue()`.  
    *Consequence*: ToolNode raises an error, LLM tries calling the exact same misspelled tool again, looping until `recursion_limit` failure.  
    *Fix*: Enforce strict Pydantic schemas and structured tool binding (`model.bind_tools(...)`).
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Blind Plan Execution Without Dynamic Re-Planning")
    st.markdown("""
    *Mechanism*: Step 1 of a 4-step plan fails completely, but the executor continues blindly executing steps 2, 3, and 4.  
    *Consequence*: Wastes API tokens on meaningless downstream executions that produce garbage output.  
    *Fix*: Implement a conditional check after each step to route to a Re-Planner if a step fails.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Engineering Simple Tasks with Plan-and-Solve")
    st.markdown("""
    *Mechanism*: Invoking a 3-node Planner-Executor graph for a simple question like "What is the capital of Japan?".  
    *Consequence*: Increases latency from 400ms to 4.5s and increases token consumption by 500%.  
    *Fix*: Use an upfront triage router to send simple queries directly to a zero-shot prompt node.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Subjective Drift in Unanchored Critic Nodes")
    st.markdown("""
    *Mechanism*: Critic node prompt lacks concrete rubrics and gives vague opinions like "make it punchier".  
    *Consequence*: Generator and reviser oscillate indefinitely without converging on a stable solution.  
    *Fix*: Provide explicit binary rubrics (e.g. "Contains type hints: Yes/No", "Word count <= 200: Yes/No").
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**ReAct**\\nA reasoning and action loop where an LLM calls tools, observes outputs, and reasons iteratively.")
with col_fc2:
    st.info("**Plan-and-Solve**\\nA pattern decoupling upfront goal planning from sequential task execution.")
with col_fc3:
    st.info("**Critic Node**\\nA specialized evaluator node providing natural language critique notes to guide revisions.")
with col_fc4:
    st.info("**Agentic RAG**\\nA retrieval pattern featuring dynamic query rewriting, document relevance grading, and fallback paths.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the eight core generative AI agent design patterns.", value=True)
c2 = st.checkbox("I know when to choose ReAct versus Plan-and-Solve.", value=True)
c3 = st.checkbox("I can implement a complete Plan-and-Execute workflow in LangGraph.", value=True)
c4 = st.checkbox("I understand how the Critic node guides revisions in Generator-Critic-Reviser loops.", value=True)
c5 = st.checkbox("I know how to prevent infinite ReAct loops using strict tool schemas.", value=True)
c6 = st.checkbox("I can design Agentic RAG workflows with dynamic relevance grading and fallback paths.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 9.2 'AI Agent Patterns' with full architectural and practical comprehension!")
