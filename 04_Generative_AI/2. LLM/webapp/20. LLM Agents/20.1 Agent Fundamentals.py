import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.1 Agent Fundamentals - LLM Engineering",
    page_icon="🤖",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class AutonomousReActAgent:
    """
    Self-contained in-memory autonomous ReAct agent implementing:
    - Thought-Action-Observation cognitive loop
    - Step budget enforcement and termination logic
    - Environment tool execution (Calculator, Knowledge Base, Search)
    - Self-reflection scratchpad state management
    """
    
    TOOLS = {
        "calculator": lambda expr: str(eval(expr, {"__builtins__": {}}, {})),
        "company_lookup": lambda query: {
            "apple": "Apple Inc. (AAPL): Q3 Revenue = 85.8B USD, Services Revenue = 24.2B USD.",
            "microsoft": "Microsoft Corp (MSFT): Q4 Revenue = 64.7B USD, Cloud Revenue = 36.8B USD.",
            "nvidia": "NVIDIA Corp (NVDA): Q2 Revenue = 30.0B USD, Data Center = 26.3B USD."
        }.get(query.lower().strip(), f"No record found for '{query}'"),
        "currency_converter": lambda amount_usd: f"{float(amount_usd) * 0.92:.2f} EUR (Rate: 1 USD = 0.92 EUR)"
    }
    
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.scratchpad: List[Dict[str, str]] = []
        
    def execute_plan(self, goal: str, planned_steps: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        self.scratchpad = []
        traces = []
        
        for idx, step in enumerate(planned_steps):
            if idx >= self.max_steps:
                traces.append({
                    "step": idx + 1,
                    "thought": "Step budget exceeded. Aborting further execution.",
                    "action": "HALT",
                    "observation": "Maximum iteration limit reached.",
                    "reflection": "Goal incomplete due to step budget cutoff."
                })
                break
                
            thought = step.get("thought", "")
            tool_name = step.get("tool", "")
            tool_arg = step.get("arg", "")
            
            # Execute Action
            if tool_name in self.TOOLS:
                try:
                    observation = self.TOOLS[tool_name](tool_arg)
                    reflection = f"Successfully obtained data from {tool_name}. Progressing towards goal."
                except Exception as err:
                    observation = f"Tool Execution Error: {str(err)}"
                    reflection = f"Action failed on {tool_name}. Need self-correction or parameter adjustment."
            else:
                observation = f"Error: Tool '{tool_name}' does not exist in registry."
                reflection = "Invalid tool selection. Recovery needed."
                
            record = {
                "step": idx + 1,
                "thought": thought,
                "action": f"{tool_name}(arg='{tool_arg}')",
                "observation": str(observation),
                "reflection": reflection
            }
            self.scratchpad.append(record)
            traces.append(record)
            
        return traces

# --- TOP-LEVEL HEADER ---
st.title("20.1 Agent Fundamentals: Cognitive Architecture, ReAct Loop, & Autonomous Reasoning")
st.caption("Chapter 20: LLM Agents | Master Architecture Specification")
st.markdown("""
A standard foundation model is a passive, stateless next-token predictor. Embedded inside an **Agent Architecture**, 
the LLM serves as an active **Cognitive Reasoning Engine**. The agent interacts with an external environment by 
formulating structured hypotheses, decomposing ambiguous prompts into executable multi-step plans, invoking external actuators 
(APIs, databases, Python sandboxes), observing real-world feedback, and engaging in **self-reflection** to adapt its strategy.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: The Autonomous Agent Cognitive Loop", divider="rainbow")

st.markdown("""
An autonomous agent operates on a continuous perception-cognition-action-reflection state machine. 
Below is the master operational blueprint showing how goals, reasoning traces, environmental actuators, and memory stores interlock.
""")

dot_blueprint = """
digraph AgentBlueprint {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_perception {
        label = "Layer 1: Perception & Ingestion";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        UserPrompt [label="User Objective / Goal\\n[High-Level Ambiguous Intent]", fillcolor="#1E3A8A", color="#3B82F6"];
        ContextEnv [label="Sensory Context & State\\n[Chat History, System Prompt, Runtimes]", fillcolor="#1E3A8A", color="#3B82F6"];
    }

    subgraph cluster_cognition {
        label = "Layer 2: Cognitive Reasoning Core (LLM Brain)";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        Planner [label="1. Goal Decomposition\\n[Single-Path, ToT, or ReAct]", fillcolor="#78350F", color="#D97706"];
        ThoughtEngine [label="2. Internal Thought Synthesis\\n[Hypothesis Formulation & Scratchpad]", fillcolor="#78350F", color="#D97706"];
        ActionSelector [label="3. Action Selection & Schema Generator\\n[Tool Calling & Parameter Validation]", fillcolor="#831843", color="#EC4899"];
        Reflector [label="5. Self-Reflection & Critique\\n[Outcome Verification & Backtracking]", fillcolor="#14532D", color="#22C55E"];
    }

    subgraph cluster_environment {
        label = "Layer 3: Environment Actuators & Memory";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#10B981";
        ToolExec [label="Actuators / Tool Dispatch\\n[Python REPL, SQL, REST APIs, Shell]", fillcolor="#312E81", color="#6366F1"];
        ObservationBus [label="4. Observation Bus\\n[Raw Execution Logs, HTTP Codes, Payloads]", fillcolor="#064E3B", color="#10B981"];
        MemoryStore [label="Short- & Long-Term Memory\\n[Vector Store + Working Buffer]", shape=cylinder, fillcolor="#1E293B", color="#38BDF8"];
    }

    UserPrompt -> Planner;
    ContextEnv -> Planner;
    Planner -> ThoughtEngine;
    ThoughtEngine -> ActionSelector;
    ActionSelector -> ToolExec [label="Dispatches Tool Call"];
    ToolExec -> ObservationBus [label="Returns Environment Result"];
    ObservationBus -> Reflector [label="Feeds Observation"];
    ObservationBus -> MemoryStore [label="Persists Trace"];
    MemoryStore -> ThoughtEngine [label="Context Ingestion"];
    Reflector -> ThoughtEngine [label="Self-Correction / Next Step", style=dashed, color="#F59E0B"];
    Reflector -> ContextEnv [label="Final Verified Answer", color="#22C55E", penwidth=2.5];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - WHAT IS AN AI AGENT? ---
st.header("2. What is an AI Agent? Autonomous vs Passive LLM Systems", divider="rainbow")

st.markdown("""
A foundation model evaluated in isolation is purely **reactive**: given input tokens, it computes probability distributions 
over a fixed vocabulary. An **AI Agent** embeds this model inside an autonomous control loop endowed with sensors, state, and actuators.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Fundamental Paradigm Shift")
    st.markdown("""
    - **Passive LLM (Stateless Transformer)**: Ingests tokens, outputs tokens. Cannot verify its own assertions, cannot interact with systems, and has no temporal permanence.
    - **Active AI Agent (Cognitive System)**: Combines reasoning with **agency**. It possesses agency because it can:
      1. Issue real-world side effects (modifying databases, executing Python code, calling APIs).
      2. Perceive dynamic outcomes of its own actions.
      3. Adapt its internal plans based on errors or partial failures without human intervention.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: How New Models Follow This
    Modern frontier LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) are post-trained specifically with 
    **native tool-use tokens** (`<tool_call>`, `func_call`). Rather than parsing fragile regex strings 
    out of raw assistant completions, foundation models emit specialized AST tokens that client runtimes execute natively.
    """)

st.markdown("### Side-by-Side Architectural Contrast: Passive LLM vs Autonomous Agent")
comparison_df = pd.DataFrame([
    {"Dimension": "Interaction Model", "Passive LLM": "Single-turn prompt -> completion", "Autonomous Agent": "Multi-turn dynamic loop (ReAct / Plan-Execute)"},
    {"Dimension": "Statefulness", "Passive LLM": "Stateless across API invocations", "Autonomous Agent": "Maintains working scratchpad + vector memory"},
    {"Dimension": "Actuation & Side Effects", "Passive LLM": "None (pure text output)", "Autonomous Agent": "Arbitrary tools (APIs, Code Sandboxes, DBs)"},
    {"Dimension": "Error Handling", "Passive LLM": "Hallucinates or outputs apologies", "Autonomous Agent": "Catches exceptions, reflects, self-corrects"},
    {"Dimension": "Termination Mechanism", "Passive LLM": "End-of-sequence (EOS) token", "Autonomous Agent": "Goal satisfaction condition or step budget cutoff"}
])
st.dataframe(comparison_df, width="stretch")

st.divider()

# --- SECTION 3: TOPIC 2 - LLM AS REASONING ENGINE ---
st.header("3. LLM as Reasoning Engine: System 1 Intuition vs System 2 Deliberation", divider="rainbow")

st.markdown("""
In cognitive science, Daniel Kahneman distinguishes between **System 1** (fast, instinctual, associative pattern recognition) 
and **System 2** (slow, deliberate, analytical step-by-step reasoning). Standard LLM token generation operates as pure System 1. 
Structuring the LLM within an agent harnesses System 2 reasoning.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Transforming Next-Token Prediction into Search")
    st.markdown("""
    - **System 1 (Raw Auto-Regressive)**: The model emits tokens sequentially. If an erroneous token is generated early in the sequence, the auto-regressive property conditions all subsequent tokens on that error, leading to compounding hallucinations.
    - **System 2 (Agentic Scaffolding)**: The agent forces intermediate thought tokens onto a private scratchpad. By separating **Thought** from **Action**, the model performs tree-search, counterfactual verification, and deliberate pruning before executing irrevocable operations.
    """)

with col2:
    st.info("""
    **The Scratchpad Theorem**:
    By allocating explicit token compute to intermediate reasoning steps (Chain-of-Thought, ReAct scratchpad), 
    the effective expressivity of transformer architectures expands from class TC0 (constant-depth threshold circuits) 
    to polynomial-time computation over bounded sequence lengths.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - THE AGENT LOOP ---
st.header("4. The Agent Loop: Perception-Cognition-Action Cycle & Step Budgets", divider="rainbow")

st.markdown("""
The canonical agent execution loop governs how state transitions from perception to action. Without strict loop governance, 
agents can fall into infinite recursive traps, consume unbound compute, or repeat failing tool calls.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("State Machine Lifecycle")
    st.markdown("""
    1. **Initialize State**: Ingest User Objective `G` and initialize Working Memory `M_0 = []`.
    2. **Cognitive Step `t`**:
       - Context Assembly: Combine system instructions, tools schema, goal `G`, and history `M_{t-1}`.
       - Model Inference: Generate internal thought `T_t` and proposed action `A_t`.
    3. **Action Execution**:
       - If `A_t == TERMINATE`, extract final answer and exit.
       - If `t >= Max_Steps`, halt with step-budget timeout.
       - Else, validate parameters against JSON Schema and dispatch to actuator.
    4. **Observation**: Capture environment result `O_t`.
    5. **Reflection**: Append `(T_t, A_t, O_t)` to `M_t`. Repeat.
    """)

with col2:
    st.warning("""
    **Production Loop Pitfalls**:
    - **Action Loop Traps**: Repeating the same query with minor phrasing changes when an API returns 404.
    - **Context Window Flooding**: Unchecked tool outputs (e.g. dumping a 5MB JSON response) can instantaneously exhaust context limits.
    - **Runaway Token Costs**: An unconstrained agent loop can burn thousands of dollars in minutes on recursive reasoning failures.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - PLANNING ---
st.header("5. Planning: Decomposition, Plan-and-Solve, & Tree-of-Thoughts", divider="rainbow")

st.markdown("""
Planning is the process of mapping a high-level, under-specified objective into a sequence of actionable sub-goals. 
Effective planning prevents agents from wandering aimlessly through action spaces.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Single-Path Planning")
    st.markdown("""
    The LLM outputs a linear sequence:
    `[Sub-goal 1 -> Sub-goal 2 -> Sub-goal 3]`.
    - **Pros**: Low latency, minimal prompt overhead.
    - **Cons**: Brittle; failure in Sub-goal 1 invalidates the entire downstream plan unless dynamic re-planning is triggered.
    """)

with col2:
    st.markdown("### Plan-and-Solve (Wang et al.)")
    st.markdown("""
    Decouples the planning phase from the execution phase:
    1. **Planner LLM**: Emits an exhaustive dependency graph of tasks.
    2. **Executor LLM**: Executes each task sequentially, feeding verified outputs to downstream tasks.
    """)

with col3:
    st.markdown("### Tree-of-Thoughts (ToT)")
    st.markdown("""
    Explores multiple branching reasoning pathways:
    - Generates candidate actions at each node.
    - Uses an evaluator LLM to score node promise.
    - Employs BFS or DFS to prune unpromising branches and backtrack on failure.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - ACTION ---
st.header("6. Action: Actuators, Tool Calling, & Parameter Binding", divider="rainbow")

st.markdown("""
An action is an explicit side-effect generated by the model intended to alter or query the external environment. 
For an action to be safely executed, the model must bind natural language reasoning into strictly typed structured payloads.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Tool Binding Contract")
    st.markdown("""
    When an LLM initiates an action, it must emit:
    1. **Target Function Identifier**: Exact registered name (e.g., `execute_sql_query`).
    2. **Typed Arguments**: Validated JSON payload adhering to the parameter schema (e.g., `{"query": "SELECT count(*) FROM users;"}`).
    3. **Call Nonce / ID**: Unique tracking identifier to pair asynchronous tool executions with incoming observations.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: Constrained Decoding
    Legacy approaches used fragile regex or markdown blocks (`Action: search[...]`) that frequently hallucinated syntax. 
    Modern runtimes employ **Context-Free Grammar (CFG) Constrained Decoding** (Outlines, Guidance, SGLang, Instructor) 
    that mask invalid next-token logits at the sampling level, mathematically guaranteeing 100% valid JSON matching the function schema.
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - OBSERVATION ---
st.header("7. Observation: Environmental Feedback, Parsing, & Context Filtering", divider="rainbow")

st.markdown("""
The observation step bridges the physical world back into the model's token space. 
How raw environment responses are sanitized, truncated, and formatted directly dictates whether the agent succeeds or derails.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Observation Engineering Principles")
    st.markdown("""
    - **Output Truncation & Summarization**: Raw API or database payloads often contain thousands of lines. The observation handler must filter down to relevant fields or summarize long payloads before re-injecting them into the context window.
    - **Error Normalization**: Stack traces must be converted into clear, semantic error messages (e.g., *"Table 'users' does not contain column 'age'"*) so the model understands why the action failed.
    - **Deterministic Formatting**: Observations should be cleanly encapsulated in role-specific blocks (`<observation>...</observation>`) to prevent prompt injection from external payloads.
    """)

with col2:
    st.code("""
# Example of Sanitized Observation Ingestion
{
    "tool_call_id": "call_982341",
    "role": "tool",
    "name": "company_lookup",
    "content": "Apple Inc. (AAPL): Q3 Revenue = 85.8B USD, Services Revenue = 24.2B USD."
}
    """, language="json")

st.divider()

# --- SECTION 8: TOPIC 7 - REFLECTION ---
st.header("8. Reflection: Self-Correction, Reflexion, & Backtracking", divider="rainbow")

st.markdown("""
Without reflection, an agent that encounters a failure repeats the exact same mistake indefinitely. 
**Self-Reflection** is the metacognitive process of evaluating whether an action moved the system closer to its goal.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Reflexion Framework (Shinn et al.)")
    st.markdown("""
    Reflexion endows agents with dynamic memory of past failures:
    - **Actor**: Proposes thoughts and actions based on trajectory history.
    - **Evaluator**: Assesses the quality of the trajectory and assigns a scalar or heuristic reward.
    - **Self-Reflection Model**: When a failure occurs, generates a verbal critique:  
      *`"My SQL query failed because I used SQLite syntax on a Postgres database. In the next attempt, I must use ILIKE instead of LIKE."`*
    - **Episodic Memory**: Critiques are stored in persistent memory and pre-pended to future prompts as operational guidelines.
    """)

with col2:
    st.info("""
    **Heuristic for Safe Reflection**:
    Reflection should occur immediately following any action that generates an error or unexpected status code. 
    If the agent reflects more than 3 consecutive times without taking a new action, the supervisor must abort to prevent metacognitive infinite loops.
    """)

st.divider()

# --- SECTION 9: PRODUCTION CODE IMPLEMENTATION ---
st.header("9. Production Implementation: End-to-End Autonomous ReAct Agent", divider="rainbow")

st.markdown("""
Below is a complete, production-ready implementation of an Autonomous ReAct Agent engine. 
It features a registered tool ecosystem, typed execution traces, budget management, and self-reflection logging.
""")

st.code('''
from typing import Dict, List, Any, Callable
import json

class ProductionReActAgent:
    """
    A robust, self-contained ReAct agent framework for LLM engineering.
    """
    
    def __init__(self, max_iterations: int = 6):
        self.max_iterations = max_iterations
        self.tool_registry: Dict[str, Callable[[str], str]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def register_tool(self, name: str, func: Callable[[str], str]) -> None:
        """Registers an executable tool into the agent registry."""
        self.tool_registry[name] = func

    def step(self, thought: str, tool_name: str, tool_arg: str) -> Dict[str, Any]:
        """Executes a single ReAct step, capturing action and observation."""
        if tool_name not in self.tool_registry:
            obs = f"Error: Tool '{tool_name}' not registered. Available tools: {list(self.tool_registry.keys())}"
            reflection = "Action failed due to unmapped tool identifier. Need tool selection correction."
        else:
            try:
                obs = self.tool_registry[tool_name](tool_arg)
                reflection = f"Tool '{tool_name}' succeeded. Evidence acquired."
            except Exception as e:
                obs = f"Execution Exception in {tool_name}: {str(e)}"
                reflection = "Encountered runtime error. Require parameter adjustment or alternative tool."

        step_record = {
            "thought": thought,
            "action": f"{tool_name}({tool_arg})",
            "observation": obs,
            "reflection": reflection
        }
        self.execution_history.append(step_record)
        return step_record

    def run_trajectory(self, goal: str, planned_steps: List[Dict[str, str]]) -> Dict[str, Any]:
        """Executes an agent trajectory with step budget enforcement."""
        self.execution_history = []
        for i, step_data in enumerate(planned_steps):
            if i >= self.max_iterations:
                return {
                    "status": "BUDGET_EXCEEDED",
                    "goal": goal,
                    "steps": self.execution_history,
                    "final_answer": "Execution stopped: Step budget reached."
                }
            self.step(
                thought=step_data.get("thought", ""),
                tool_name=step_data.get("tool", ""),
                tool_arg=step_data.get("arg", "")
            )
        return {
            "status": "COMPLETED",
            "goal": goal,
            "steps": self.execution_history,
            "final_answer": "Goal successfully fulfilled."
        }
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Agent Workbench: Step-by-Step ReAct Execution Simulator", divider="rainbow")

st.markdown("""
Experiment with an active ReAct autonomous agent. Select a financial research goal, customize the step budget, 
and observe how the agent interleaves **Thoughts**, **Actions**, **Observations**, and **Reflections**.
""")

col1, col2 = st.columns([1, 2])

with col1:
    selected_goal = st.selectbox(
        "Select User Objective / Goal:",
        [
            "Calculate Apple vs Microsoft combined Q3/Q4 revenue in EUR",
            "Lookup NVIDIA Data Center revenue and convert to EUR",
            "Execute Invalid Tool to test Agent Self-Correction"
        ]
    )
    
    step_budget = st.slider("Step Budget (Max Iterations):", min_value=1, max_value=6, value=4)
    run_btn = st.button("🚀 Run Autonomous Agent Trajectory", type="primary", use_container_width=True)

with col2:
    if run_btn:
        agent = AutonomousReActAgent(max_steps=step_budget)
        
        # Configure simulated plans based on selected goal
        if "Apple vs Microsoft" in selected_goal:
            plan = [
                {"thought": "I need to query Apple's latest quarterly revenue.", "tool": "company_lookup", "arg": "apple"},
                {"thought": "I need to query Microsoft's latest quarterly revenue.", "tool": "company_lookup", "arg": "microsoft"},
                {"thought": "Sum Apple (85.8B USD) and Microsoft (64.7B USD) revenues: 85.8 + 64.7 = 150.5B.", "tool": "calculator", "arg": "85.8 + 64.7"},
                {"thought": "Convert the total 150.5B USD into EUR at current rates.", "tool": "currency_converter", "arg": "150.5"}
            ]
        elif "NVIDIA" in selected_goal:
            plan = [
                {"thought": "I need to lookup NVIDIA's financial report for Data Center numbers.", "tool": "company_lookup", "arg": "nvidia"},
                {"thought": "Convert Data Center revenue ($26.3B USD) into EUR.", "tool": "currency_converter", "arg": "26.3"}
            ]
        else:
            plan = [
                {"thought": "Attempting to query database using an unregistered tool.", "tool": "legacy_sql_tool", "arg": "SELECT * FROM sales;"},
                {"thought": "Fallback to registered company_lookup tool after failure.", "tool": "company_lookup", "arg": "apple"}
            ]
            
        traces = agent.execute_plan(selected_goal, plan)
        
        st.subheader("Autonomous Trajectory Trace")
        for t in traces:
            with st.expander(f"📍 Step {t['step']}: {t['action']}", expanded=True):
                st.markdown(f"**💭 Thought**: `{t['thought']}`")
                st.markdown(f"**⚡ Action**: `{t['action']}`")
                st.markdown(f"**👁️ Observation**: `{t['observation']}`")
                st.markdown(f"**🔄 Reflection**: `{t['reflection']}`")
                
        st.success(f"Execution completed. Total steps utilized: {len(traces)} of budget {step_budget}.")
    else:
        st.info("Select an objective and click 'Run Autonomous Agent Trajectory' to observe live execution.")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: What distinguishes the ReAct paradigm from Plan-and-Solve, and when should you choose each?"):
    st.markdown("""
    **Answer**:
    - **ReAct** interleaves reasoning with immediate action execution step-by-step (`Thought -> Action -> Observation -> Thought`). It is optimal for **dynamic, uncertain environments** where each step's observation dramatically alters what should happen next (e.g. debugging, web exploration).
    - **Plan-and-Solve** completely decouples planning from execution. An initial LLM generates an entire task dependency graph, and worker agents execute the sub-tasks (often in parallel). It is optimal for **predictable, composable multi-step tasks** where sub-tasks are independent and high throughput is required.
    """)

with st.expander("Q2: How do you prevent an autonomous agent from entering an infinite action loop?"):
    st.markdown("""
    **Answer**:
    Production agent frameworks enforce three complementary guardrails:
    1. **Strict Step Budget**: Hard ceiling on max iterations (e.g., 10 steps).
    2. **State Repetition Detection**: Computing hashes of `(tool_name, arguments)` across the scratchpad. If the exact same call is repeated 2+ times with equivalent observations, force an interrupt or trigger an explicit replanning prompt.
    3. **Token / Cost Circuit Breakers**: Halting execution if cumulative API spend for the session breaches a monetary threshold.
    """)

with st.expander("Q3: Why are modern foundation models migrating away from parsing raw text actions to native tool calling tokens?"):
    st.markdown("""
    **Answer**:
    Parsing unstructured strings (`Action: tool_name[arg]`) suffers from:
    1. **High Parsing Failure Rates**: Models often insert conversational filler, omit brackets, or format arguments inconsistently.
    2. **Context Waste**: String delimiters consume extra token context.
    3. **Grammar Enforcement**: Native tool calling tokens (`<tool_call>`) enable runtime constrained decoding (CFG masks), ensuring zero JSON syntax errors and strict adherence to registered OpenAPI/JSON Schema definitions.
    """)

with st.expander("Q4: How does Reflexion differ from standard ReAct error handling?"):
    st.markdown("""
    **Answer**:
    In standard ReAct, when a tool returns an error, the agent sees the error in its immediate working memory and attempts a correction in the next token. 
    In **Reflexion**, the entire failed trajectory is reviewed by an Evaluator model, which generates a natural language self-critique. 
    This critique is stored in a **persistent episodic memory store**, allowing the agent to remember its failure modes across completely distinct future sessions and user queries.
    """)

with st.expander("Q5: What is the Context Window Flooding vulnerability in agent systems, and how is it mitigated?"):
    st.markdown("""
    **Answer**:
    Context Window Flooding occurs when an actuator returns an unexpectedly massive observation (e.g., a SQL query returning 50,000 rows or a web scraper returning an entire HTML dump), instantaneously overflowing the context window or diluting needle-in-a-haystack attention. 
    Mitigation strategies include:
    1. Truncating raw observations to a hard character limit.
    2. Routing large observations through a fast summarization model before appending to the scratchpad.
    3. Storing raw tool outputs in external object storage (S3) and passing only a URI + schema summary to the agent.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Runaway Recursive Tool Loops**
    Allowing an agent to execute without a hard step budget and loop detection. When an external API changes its schema or returns persistent 500 errors, an unconstrained agent can burn hundreds of dollars in API credits within minutes.
    """)
    st.error("""
    **TRAP 2 (Error): Unsanitized Code Execution Sandboxes**
    Allowing an agent to run arbitrary Python code via `exec()` or `eval()` without containerized network and filesystem isolation, exposing the host machine to command injection or data exfiltration.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Tool Output Dilution**
    Dumping raw, unformatted JSON responses from APIs directly into the agent scratchpad. This clutters the context window, accelerates attention degradation, and drastically increases prompt prefill latency.
    """)
    st.warning("""
    **TRAP 4 (Warning): Blind Tool Execution Without Schema Validation**
    Passing model-generated parameters directly into backend drivers without Pydantic/JSON Schema validation, causing silent type coercion bugs or unexpected database corruption.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **System 1 vs 2**  
    *System 1*: Reactive next-token prediction.  
    *System 2*: Scaffolding with scratchpads, search, and deliberate self-critique.
    """)
with fc2:
    st.markdown("""
    **ReAct Pattern**  
    Couples internal reasoning traces (Thoughts) with external actions (Actions) and environment feedback (Observations).
    """)
with fc3:
    st.markdown("""
    **Step Budget**  
    The maximum iteration threshold allocated to an agent run to prevent runaway compute and recursive traps.
    """)
with fc4:
    st.markdown("""
    **Reflexion**  
    Metacognitive critique stored in episodic memory to prevent repeating identical failures across future tasks.
    """)

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist: Agent Fundamentals Mastery", divider="rainbow")

st.markdown("Verify your understanding of autonomous agent foundations by reviewing each criterion:")

chk1 = st.checkbox("I understand the architectural difference between a passive stateless LLM and an autonomous cognitive agent.", key="af_chk1")
chk2 = st.checkbox("I can trace the mathematical and state transitions of the Thought-Action-Observation ReAct cycle.", key="af_chk2")
chk3 = st.checkbox("I know how System 2 deliberation expands transformer expressivity via intermediate scratchpad tokens.", key="af_chk3")
chk4 = st.checkbox("I can differentiate Single-Path Planning, Plan-and-Solve, and Tree-of-Thoughts (ToT) architectures.", key="af_chk4")
chk5 = st.checkbox("I understand how observation truncation and error normalization prevent context window flooding.", key="af_chk5")
chk6 = st.checkbox("I know how the Reflexion framework uses episodic memory and critiques to enable permanent cross-session self-correction.", key="af_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of Agent Fundamentals and Cognitive Architectures!")
