import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="22.1 Planning Agents - LangGraph Curriculum",
    page_icon="📋",
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
    .badge-plan {
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
class PlanningAgentsSimulator:
    """
    Simulates Plan-and-Execute workflows in LangGraph:
    - High-level task decomposition
    - Step-by-step sequential execution
    - Dynamic re-planning triggered by failed steps
    """
    def __init__(self):
        pass

    def execute_plan(self, goal: str, inject_failure: bool) -> Dict[str, Any]:
        initial_plan = [
            {"step": 1, "task": f"Analyze requirements for: {goal}", "status": "COMPLETED"},
            {"step": 2, "task": "Query internal corporate databases for records", "status": "PENDING"},
            {"step": 3, "task": "Generate synthesis report and verify numbers", "status": "PENDING"}
        ]

        trace = []
        trace.append({"step": 1, "action": "Planner initialized 3-step execution plan."})

        if not inject_failure:
            initial_plan[1]["status"] = "COMPLETED"
            initial_plan[2]["status"] = "COMPLETED"
            trace.append({"step": 2, "action": "Step 2 executed successfully."})
            trace.append({"step": 3, "action": "Step 3 executed successfully. Final answer produced."})
            final_status = "SUCCESS"
        else:
            initial_plan[1]["status"] = "FAILED (Database Connection Timeout)"
            trace.append({"step": 2, "action": "Step 2 failed! Triggered Replanner node."})
            # Dynamic Plan Revision
            revised_plan = [
                {"step": 1, "task": initial_plan[0]["task"], "status": "COMPLETED"},
                {"step": 2, "task": "Query secondary read-replica cache (Replanned)", "status": "COMPLETED"},
                {"step": 3, "task": initial_plan[2]["task"], "status": "COMPLETED"}
            ]
            trace.append({"step": 2.5, "action": "Replanner revised Step 2 to target replica cache."})
            trace.append({"step": 3, "action": "Step 3 executed successfully with replica data."})
            initial_plan = revised_plan
            final_status = "RECOVERED_VIA_REPLANNING"

        return {
            "initial_goal": goal,
            "final_plan": initial_plan,
            "trace": trace,
            "status": final_status
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Plan-and-Solve Topologies, State Staging, and Dynamic Plan Revision", divider="rainbow")

st.markdown("""
Standard ReAct agents make decisions one step at a time: they see a prompt, call a tool, inspect the result, and decide the next tool.
For long-horizon tasks (writing a comprehensive research report, refactoring a multi-file repository), ReAct agents frequently get stuck in local minima, lose track of the overall goal, or wander down infinite rabbit holes.

**Planning Agents (Plan-and-Solve Architecture)** decouple deliberative planning from tactical execution:
1. **Planner Node**: Formulates a complete multi-step execution plan before executing any tools.
2. **Executor Node**: Executes the current step in the plan using specialized tools.
3. **Replanner Node**: Inspects execution outputs, updates completed steps, and dynamically adjusts the remaining plan if obstacles or new discoveries occur.

The blueprint below illustrates this cyclic planning and replanning architecture.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph PlanningArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_planning {
            label="1. Deliberative Planning Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_goal [label="User Goal\\n'Migrate User Service from Flask to FastAPI'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            planner_node [label="Planner Node\\nDecomposes into Ordered Sub-Goals", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_execution {
            label="2. Tactical Execution Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            task_selector [label="Step Dispatcher\\nExtracts Next Pending Step", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            step_executor [label="Executor Node (Tools)\\nRuns Sub-Task to Completion", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
        }

        subgraph cluster_replanning {
            label="3. Reactive Replanning & Convergence Gate";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            replanner_node [label="Replanner Node\\nDid Step Succeed? Adapt Plan?", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            terminal_delivery [label="Final Response Synthesis\\nAll Sub-Goals Verified", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        user_goal -> planner_node;
        planner_node -> task_selector [label="Emits Initial Plan"];
        task_selector -> step_executor [label="Active Step"];
        step_executor -> replanner_node [label="Step Output"];

        replanner_node -> task_selector [label="Remaining Steps Pending", color="#60A5FA"];
        replanner_node -> task_selector [label="Step Failed -> Plan Revised", color="#FBBF24", style=dashed];
        replanner_node -> terminal_delivery [label="Plan Fully Executed", color="#34D399"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: PLANNING
# ==============================================================================
st.header("2. Planning: Long-Horizon Goal Formulation and Deliberative Reasoning", divider="rainbow")

st.markdown("""
**Planning** allows an AI agent to build a mental map of execution before spending tokens on expensive external tools.
- **Top-Down Perspective**: Analyzes the problem globally, identifying prerequisite dependencies.
- **Structured Schema**: Rather than returning free-form text, the planner yields a structured list of actionable steps.
- **Reduces Tool Churn**: Prevents the agent from wandering aimlessly by anchoring execution to explicit milestone criteria.
""")

# ==============================================================================
# SECTION 3: TASK DECOMPOSITION
# ==============================================================================
st.header("3. Task Decomposition: Breaking Monolithic Objectives into Bounded Sub-Goals", divider="rainbow")

st.markdown("""
Complex tasks like *"Perform a competitor security audit"* cannot be executed in one shot.
**Task Decomposition** partitions the objective into modular, bite-sized tasks:
1. Enumerate target public subdomains.
2. Probe SSL certificate expirations and cipher suites.
3. Audit open DNS records and SPF/DMARC configurations.
4. Synthesize risk score matrix.
""")

# ==============================================================================
# SECTION 4: SUBTASK GENERATION
# ==============================================================================
st.header("4. Subtask Generation: Structured Step Schemas and Dependency Graph Assembly", divider="rainbow")

st.markdown("""
Each subtask in the plan is modeled using a strongly-typed schema:
""")

st.code('''from pydantic import BaseModel, Field
from typing import List

class SubTask(BaseModel):
    step_id: int
    description: str
    expected_output: str
    dependencies: List[int] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    steps: List[SubTask]''', language="python")

# ==============================================================================
# SECTION 5: TASK EXECUTION
# ==============================================================================
st.header("5. Task Execution: Iterative Step Execution and State Progress Tracking", divider="rainbow")

st.markdown("""
The **Executor Node** focuses strictly on the active pending step.
- Isolates context: The executor does not need to worry about steps 4 and 5; it concentrates only on completing step 2.
- Upon completion, it appends the intermediate result to an `execution_results` state channel.
""")

# ==============================================================================
# SECTION 6: PLAN REVISION
# ==============================================================================
st.header("6. Plan Revision: Reactive Replanning Based on Execution Observations and Tool Feedback", divider="rainbow")

st.markdown("""
Static plans frequently break when confronted with real-world environments:
- Step 2 discovers that an API endpoint is deprecated.
- Step 3 returns unexpected data that invalidates Step 4.

The **Replanner Node** evaluates the execution history and decides:
1. Mark current step as complete.
2. Remove steps that are no longer necessary.
3. Insert new recovery or investigation steps into the plan.
4. Conclude if the ultimate goal has already been achieved early.
""")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full Plan-and-Execute Agent Graph with Dynamic Re-Planning", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a Plan-and-Execute state machine in LangGraph.
""")

st.code('''from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class PlanExecuteState(TypedDict):
    input_goal: str
    plan: List[str]
    past_steps: List[tuple]
    response: str

def planner_node(state: PlanExecuteState) -> dict:
    goal = state["input_goal"]
    # Formulate initial decomposition
    initial_plan = [
        f"1. Research background on {goal}",
        f"2. Synthesize key arguments for {goal}",
        f"3. Generate final executive brief for {goal}"
    ]
    return {"plan": initial_plan}

def executor_node(state: PlanExecuteState) -> dict:
    current_step = state["plan"][0]
    # Execute current step with tools
    result = f"Result of executing [{current_step}]"
    return {
        "past_steps": state.get("past_steps", []) + [(current_step, result)]
    }

def replanner_node(state: PlanExecuteState) -> dict:
    remaining_steps = state["plan"][1:] # Pop completed step
    if not remaining_steps:
        return {
            "plan": [],
            "response": f"Successfully completed all tasks for: {state['input_goal']}"
        }
    return {"plan": remaining_steps}

def route_plan(state: PlanExecuteState) -> str:
    if not state["plan"]:
        return END
    return "executor"

builder = StateGraph(PlanExecuteState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("replanner", replanner_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "replanner")
builder.add_conditional_edges("replanner", route_plan, {"executor": "executor", END: END})

app = builder.compile(checkpointer=MemorySaver())
print("Plan-and-Execute Agent StateGraph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Plan Decomposition & Replanning Simulator", divider="rainbow")

st.markdown("""
Test how planning agents break goals into subtasks and dynamically adapt plans when failures occur.
""")

sim = PlanningAgentsSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Mission Goal Configuration")
    user_goal = st.text_input("High-Level Goal", value="Audit Q3 Financial Statements & Verify Reserves")
    simulate_failure = st.toggle("Simulate Step 2 Database Failure (Triggers Re-Planning)", value=True)

    if st.button("🚀 Execute Planning Workflow"):
        res = sim.execute_plan(user_goal, simulate_failure)
        st.session_state["plan_sim_res"] = res

with col_sim2:
    st.subheader("📋 Plan State & Execution Trace")
    if "plan_sim_res" in st.session_state:
        r = st.session_state["plan_sim_res"]
        
        st.info(f"Execution Outcome: **{r['status']}**")
        st.markdown("#### Plan Milestone Progress:")
        st.dataframe(pd.DataFrame(r["final_plan"]), use_container_width=True)
        
        st.markdown("#### Event Lifecycle Log:")
        for ev in r["trace"]:
            st.markdown(f"- **Step {ev['step']}**: {ev['action']}")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: How do Plan-and-Solve agents fundamentally differ from standard ReAct agents?", expanded=False):
    st.markdown("""
    **Answer**:
    ReAct agents decide one tool step at a time, making them short-sighted and prone to circular loops on long tasks.
    Plan-and-Solve agents generate a structured, comprehensive execution plan first, execute subtasks systematically, and update the plan only when execution feedback warrants replanning.
    """)

with st.expander("Q2: What is the primary role of the Replanner node in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    The Replanner node evaluates the output of the most recent step against the overall goal.
    It pops completed steps, updates the remaining plan, inserts recovery steps if a failure occurred, and terminates execution once the goal is satisfied.
    """)

with st.expander("Q3: How do planning agents prevent infinite replanning loops?", expanded=False):
    st.markdown("""
    **Answer**:
    By maintaining a `replanning_iterations` counter in the state schema. If the plan is revised more than $N$ times (e.g. 3 times) without progress, the graph halts and escalates to a human.
    """)

with st.expander("Q4: Can subtasks in a plan be executed in parallel?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! If the planner marks subtasks as independent (no shared dependencies), the graph can use the `Send` API or parallel nodes to execute multiple subtasks concurrently in the same superstep.
    """)

with st.expander("Q5: Why should the Planner and Executor be separate LLM calls?", expanded=False):
    st.markdown("""
    **Answer**:
    Separation allows using different specialized models: a larger, highly capable reasoning model for planning (e.g. Claude 3.5 Sonnet / GPT-4o), and smaller, faster, cheaper models for executing focused subtasks (e.g. GPT-4o-mini).
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Replanning Loops on Persistent Failures")
    st.markdown("""
    *Mechanism*: Replanner continuously regenerating new plans when an external database is completely down.  
    *Consequence*: Burns hundreds of thousands of tokens cycling between executor and replanner.  
    *Fix*: Enforce a hard ceiling on `max_replan_iterations <= 3`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Executor Re-Running Completed Steps")
    st.markdown("""
    *Mechanism*: Failing to pop completed steps from `state['plan']`.  
    *Consequence*: The executor repeats Step 1 endlessly in an infinite loop.  
    *Fix*: Always slice `state['plan'][1:]` or maintain an explicit completed list.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Over-Planning for Trivial Single-Step Tasks")
    st.markdown("""
    *Mechanism*: Invoking a 3-step planning graph for "What is 2 + 2?".  
    *Consequence*: Adds 3 unnecessary LLM calls and 4 seconds of latency to a simple question.  
    *Fix*: Use an ingress router to bypass the planning graph for simple questions.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Cost Warning): Passing Entire Execution History to the Planner")
    st.markdown("""
    *Mechanism*: Passing 50KB of raw tool outputs into the replanner prompt.  
    *Consequence*: Causes token bloat and distracts the replanner from the high-level roadmap.  
    *Fix*: Summarize past step results into concise 1-sentence completions.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Plan-and-Solve**\\nA pattern decoupling deliberative global planning from tactical step execution.")
with col_fc2:
    st.info("**Replanner Node**\\nA node dynamically modifying remaining steps based on intermediate tool outputs.")
with col_fc3:
    st.info("**Task Decomposition**\\nBreaking large complex goals into small, verifiable, sequential sub-tasks.")
with col_fc4:
    st.info("**Deliberative Reasoning**\\nFormulating an overarching execution strategy before spending tool tokens.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural differences between Plan-and-Solve and ReAct.", value=True)
c2 = st.checkbox("I know how to structure an ExecutionPlan schema with Pydantic and TypedDict.", value=True)
c3 = st.checkbox("I understand how the Replanner node adapts remaining tasks dynamically.", value=True)
c4 = st.checkbox("I know how to prevent infinite replanning loops with iteration guards.", value=True)
c5 = st.checkbox("I can implement a complete Plan-and-Execute cyclic graph in LangGraph.", value=True)
c6 = st.checkbox("I understand when to bypass planning graphs for simple single-step queries.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 22.1 'Planning Agents' with full architectural and practical comprehension!")
