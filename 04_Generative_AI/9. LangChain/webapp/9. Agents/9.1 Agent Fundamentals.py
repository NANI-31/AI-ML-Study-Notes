import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="9.1 Agent Fundamentals - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("9.1 Agent Fundamentals: Autonomous Reasoning, ReAct Loops & Tool Decoupling")
st.caption("Chapter 9: Agents | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In classical pipelines and LCEL chains, the execution sequence is predetermined at authoring time: 
Step A feeds into Step B, which feeds into Step C. 

In contrast, an **AI Agent** uses a Large Language Model as an **autonomous reasoning engine**. 
Given a high-level goal, the agent dynamically decides *which* steps to execute, *which* tools to invoke, 
and *when* enough factual evidence has been collected to terminate and deliver a final answer.

In this lesson, you will master:
- **What is an Agent?**: Autonomous entities operating over perception, reasoning, and actuation.
- **Agent vs Chain**: The architectural divergence between deterministic DAGs and dynamic cyclic state machines.
- **The Agent Loop**: The iterative sense-reason-act cycle governing execution.
- **Reasoning and Action (ReAct)**: The canonical interleaving of *Thought*, *Action*, and *Observation*.
- **Tool Selection**: How models choose tools based on semantic prompt descriptions and context.
- **Tool Execution**: Safe host dispatch and observation ingestion.
- **Final Response Generation**: Identifying terminal stopping conditions and synthesizing grounded conclusions.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockAgentStepRecord:
    """Represents a single step in a ReAct cognitive trace in module scope."""
    def __init__(self, step_number: int, thought: str, action: str, action_input: Dict[str, Any], observation: str):
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation

class MockReActAgentSimulator:
    """Simulates an autonomous multi-turn ReAct reasoning loop in module scope."""
    def __init__(self):
        pass

    def run_agent_task(self, task_prompt: str, max_turns: int) -> Dict[str, Any]:
        trace: List[MockAgentStepRecord] = []
        task_lower = task_prompt.lower()

        # Step 1: Initial Investigation
        thought_1 = f"To address '{task_prompt}', I first need to verify the user account status in the customer database."
        act_1 = "query_customer_db"
        input_1 = {"user_email": "ops-lead@enterprise.corp"}
        obs_1 = "User Record: ID=8492, Status=ACTIVE, Tier=ENTERPRISE, OpenIncidents=1."
        trace.append(MockAgentStepRecord(1, thought_1, act_1, input_1, obs_1))

        # Step 2: Second Reason-Act cycle
        if "latency" in task_lower or "performance" in task_lower or "slow" in task_lower:
            thought_2 = "The account is valid and active. Now I must query cluster telemetry for reported latency spikes."
            act_2 = "query_cluster_metrics"
            input_2 = {"service": "api-gateway", "metric": "p99_latency_ms"}
            obs_2 = "Telemetry Report: p99 latency spiked to 1,240ms between 14:00 and 14:15 UTC due to DB pool exhaustion."
            trace.append(MockAgentStepRecord(2, thought_2, act_2, input_2, obs_2))
            
            final_answer = (
                "Final Resolution: Customer account #8492 (Enterprise Tier) experienced elevated p99 latency "
                "(1,240ms) due to database connection pool exhaustion between 14:00 and 14:15 UTC. "
                "Database pool size has been scaled to 50 connections and latency has stabilized at 42ms."
            )
        else:
            thought_2 = "The account is active. Now checking active support tickets and open incident logs."
            act_2 = "fetch_support_ticket"
            input_2 = {"ticket_id": "INC-9481"}
            obs_2 = "Ticket Details: INC-9481: Scheduled maintenance completed successfully. All replicas healthy."
            trace.append(MockAgentStepRecord(2, thought_2, act_2, input_2, obs_2))
            
            final_answer = (
                "Final Resolution: Investigated customer record #8492. All active tickets are resolved "
                "following successful scheduled maintenance, and all cluster replicas are verified healthy."
            )

        return {
            "total_turns": len(trace),
            "trace": trace,
            "final_answer": final_answer,
            "status": "GOAL_REACHED"
        }

# --- SECTION 1: ARCHITECTURE BLUEPRINT & COGNITIVE AGENT TOPOLOGY ---
st.header("1. Architecture Blueprint: The Autonomous Cognitive Agent Architecture", divider="rainbow")

st.markdown("""
The architecture below details the cyclic cognitive topology of a modern AI Agent. 
Unlike linear chains that progress forward only, the agent maintains an **iterative state loop**: 
evaluating whether the current state satisfies the terminal condition or requires further tool execution.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Tier 1: User Goal & Initial State
        subgraph cluster_intake {
            label = "1. OBJECTIVE &amp; INITIAL STATE TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserGoal [label="User Task Objective\\n'Diagnose why customer checkout is failing'"];
            AgentState [label="Agent Conversation State\\nMessages: [SystemMessage, HumanMessage]"];
        }

        // Tier 2: Reasoning Core (The Mind)
        subgraph cluster_mind {
            label = "2. REASONING &amp; DECISION ENGINE (LLM)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            ReasoningNode [label="Model Autoregressive Generation\\nEvaluates goal vs existing observations\\nDecides Next Cognitive Step", color="#10B981"];
            DecisionBranch [label="Decide Next Action:\\nHas sufficient evidence been gathered?"];
        }

        // Tier 3: Action & Tools (The Hands)
        subgraph cluster_actuation {
            label = "3. ACTUATION &amp; TOOL DISPATCH TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ToolCallNode [label="Emit ToolCall(name='query_db', args={...})\\nSelects specific tool from tool registry"];
            ToolExecutor [label="Host Runtime Tool Execution\\nRuns physical Python / API function"];
            ObservationNode [label="ToolMessage(content='Observation...')\\nCaptures real-world factual result", color="#38BDF8"];
        }

        // Tier 4: Terminal Delivery
        subgraph cluster_delivery {
            label = "4. RESOLUTION &amp; DELIVERY TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            FinalAnswerNode [label="Final AIMessage Synthesis\\nDelivers grounded explanation to user", color="#10B981"];
        }

        // Wiring
        UserGoal -> AgentState;
        AgentState -> ReasoningNode;
        ReasoningNode -> DecisionBranch;

        // Cyclic Loop
        DecisionBranch -> ToolCallNode [label="Need more info / Tool required"];
        ToolCallNode -> ToolExecutor;
        ToolExecutor -> ObservationNode;
        ObservationNode -> AgentState [label="Append Observation &amp; Loop", color="#38BDF8", penwidth=2.0];

        // Terminal Exit
        DecisionBranch -> FinalAnswerNode [label="Goal Satisfied (Stop Condition)", color="#10B981", penwidth=2.0];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Architectural Insight:**
The defining characteristic of an agent is the **Observation ➔ State ➔ Reason cyclic feedback loop**. 
The path taken through tools is determined at runtime based on the actual observations returned, 
not hard-coded into static execution branches.
""")

st.divider()

# --- SECTION 2: WHAT IS AN AGENT? ---
st.header("2. What Is an Agent? (Autonomous Cognitive Entities)", divider="rainbow")

st.markdown("""
In artificial intelligence literature, an **Agent** is defined as an entity that perceives its environment, 
maintains internal state, reasons about actions, and executes tools to achieve a specified goal.

In modern GenAI engineering, an agent combines:
1. **The Brain (LLM)**: Provides semantic reasoning, decomposition, and natural language synthesis.
2. **The Memory (State)**: Tracks conversational context, intermediate tool observations, and scratchpads.
3. **The Tools (Actuators)**: Grants access to real-time search, databases, APIs, and code sandboxes.
4. **The Loop (Runtime)**: Orchestrates the iterative invocation cycle until a termination condition is met.
""")

st.divider()

# --- SECTION 3: AGENT VS CHAIN ---
st.header("3. Agent vs Chain (Deterministic DAGs vs Cyclic Decision Engines)", divider="rainbow")

st.markdown("""
Understanding when to use a **Chain** versus an **Agent** is one of the most critical architectural decisions in system design.
""")

col_cmp1, col_cmp2 = st.columns(2)
with col_cmp1:
    st.info("🔗 LCEL Chain (Directed Acyclic Graph)")
    st.markdown("""
    - **Execution Flow**: Fixed, deterministic, strictly ordered.
    - **Path Decision**: Hard-coded by the developer at compile time.
    - **Control Flow**: Linear or branching DAG (`RunnableSequence`, `RunnableBranch`).
    - **Latency & Cost**: Predictable and bounded.
    - **Ideal For**: Document summarization, structured extraction, translation, static RAG.
    """)

with col_cmp2:
    st.success("🤖 AI Agent (Cyclic State Machine)")
    st.markdown("""
    - **Execution Flow**: Dynamic, non-deterministic, adaptive.
    - **Path Decision**: Decided by the LLM at runtime based on real-time observations.
    - **Control Flow**: Cyclic state machine (`Model ➔ Tool ➔ Model`).
    - **Latency & Cost**: Variable; depends on number of reasoning turns needed.
    - **Ideal For**: Autonomous troubleshooting, research assistants, multi-database query synthesis.
    """)

st.divider()

# --- SECTION 4: THE AGENT LOOP ---
st.header("4. The Agent Loop (The Sense-Reason-Act Perception Loop)", divider="rainbow")

st.markdown("""
Every AI agent operates on a continuous perception-action loop:
1. **Perceive (Sense)**: Inspect current conversation history and latest tool observations.
2. **Reason (Think)**: Evaluate remaining sub-goals against available tools.
3. **Act (Execute)**: Emit a tool call or deliver the final answer.
4. **Update (Observe)**: Ingest tool execution output into memory and repeat.
""")

loop_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=LR;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    Sense [label="1. SENSE\\nIngest history &amp; observations", fillcolor="#0F172A"];
    Think [label="2. REASON\\nLLM decides next step", fillcolor="#1E293B", color="#10B981"];
    Act [label="3. ACT\\nDispatch tool or emit text", fillcolor="#0F172A"];
    Observe [label="4. OBSERVE\\nCapture environment output", fillcolor="#1E293B", color="#38BDF8"];

    Sense -> Think -> Act -> Observe -> Sense;
}
"""
st.graphviz_chart(loop_dot, use_container_width=True)

st.divider()

# --- SECTION 5: REASONING AND ACTION (ReAct) ---
st.header("5. Reasoning and Action (The ReAct Framework)", divider="rainbow")

st.markdown("""
Introduced by Yao et al. (2022), the **ReAct (Reasoning + Acting)** framework pioneered the standard approach 
for prompting LLMs to solve complex tasks. 

Instead of jumping directly to action, the model explicitly verbalizes a **Thought**, follows it with an **Action**, 
and pauses until the environment returns an **Observation**:
""")

st.code('''# Canonical ReAct Trace Pattern:
Question: What is the age difference between the CEO of Apple and the CEO of Microsoft?

Thought 1: I need to find the birth date or age of the CEO of Apple (Tim Cook).
Action 1: search_person(name="Tim Cook")
Observation 1: Tim Cook (born November 1, 1960) is the CEO of Apple.

Thought 2: Tim Cook was born in 1960. Now I need to find the birth date of the CEO of Microsoft (Satya Nadella).
Action 2: search_person(name="Satya Nadella")
Observation 2: Satya Nadella (born August 19, 1967) is the CEO of Microsoft.

Thought 3: Tim Cook was born in 1960, and Satya Nadella was born in 1967. The age difference is 1967 - 1960 = 7 years.
Final Answer: The age difference between Tim Cook (born 1960) and Satya Nadella (born 1967) is approximately 7 years.
''', language="text")

st.markdown("""
By decomposing problems into explicit thoughts before each action, ReAct dramatically reduces hallucination 
and enables self-correction when intermediate observations reveal errors.
""")

st.divider()

# --- SECTION 6: TOOL SELECTION ---
st.header("6. Tool Selection (How Models Choose Tools)", divider="rainbow")

st.markdown("""
How does an LLM know which tool to select among 20 available options?
The model performs **semantic matching** between its intermediate reasoning goals and the tools' descriptions:
1. **Description Grounding**: The LLM reads tool descriptions in the prompt to match intent.
2. **Argument Alignment**: The LLM inspects argument names, types, and constraints in the JSON Schema.
3. **Disambiguation**: Clear descriptions with explicit boundary rules prevent the model from confusing similar tools.
""")

st.divider()

# --- SECTION 7: TOOL EXECUTION ---
st.header("7. Tool Execution (Host Runtime Dispatch & Sandboxing)", divider="rainbow")

st.markdown("""
When the LLM selects a tool, it outputs a structured `tool_calls` dictionary. 
The host application:
1. Verifies that the requested tool exists in the application's whitelist registry.
2. Validates incoming parameters against the tool's Pydantic schema.
3. Executes the underlying Python callable within safety boundaries (timeouts, rate limits, read-only guards).
4. Packages the return value into a `ToolMessage` and appends it to state history.
""")

st.divider()

# --- SECTION 8: FINAL RESPONSE GENERATION ---
st.header("8. Final Response Generation (Termination Conditions & Synthesis)", divider="rainbow")

st.markdown("""
How does an agent know when to stop? 
The agent evaluates **Termination Conditions**:
1. **Goal Satisfaction**: The model determines that all necessary facts have been acquired and emits an `AIMessage` containing plain text with **no `tool_calls`**.
2. **Maximum Iterations Safeguard (`max_iterations`)**: If the agent enters an unproductive loop, a hard limit (e.g. 10 turns) aborts execution to prevent runaway costs.
3. **Explicit Stop Token**: Special tokens or structured flags indicating completion.
""")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise ReAct Cognitive Agent Pipeline", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates an autonomous ReAct agent loop 
using modern LangChain primitives, parallel tool execution, and defensive loop bounds.
""")

st.code('''import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

logger = logging.getLogger("EnterpriseReActAgent")

# 1. Define Production Tools
@tool
def query_customer_database(email: str) -> str:
    """Lookup customer account details, tier, and open tickets by email."""
    return f"Customer '{email}': ID=4820, Tier=ENTERPRISE, OpenIncidents=['INC-9102']"

@tool
def fetch_incident_diagnostics(incident_id: str) -> str:
    """Fetch root-cause logs for an operational incident ticket."""
    return f"Incident {incident_id}: API Gateway 504 Gateway Timeout during deployment roll-out."

# 2. Build the ReAct Autonomous Controller
class ProductionReActAgent:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [query_customer_database, fetch_incident_diagnostics]
        self.tool_map = {t.name: t for t in self.tools}
        self.bound_model = self.model.bind_tools(self.tools)

    def run(self, user_goal: str) -> str:
        messages: List[BaseMessage] = [
            HumanMessage(content=user_goal)
        ]
        
        logger.info(f"Agent initiated with goal: {user_goal}")

        for turn in range(1, self.max_turns + 1):
            logger.info(f"--- Reasoning Turn {turn}/{self.max_turns} ---")
            
            # Step 1: Model Reasoning Step
            response: AIMessage = self.bound_model.invoke(messages)
            messages.append(response)

            # Step 2: Termination Check
            if not response.tool_calls:
                logger.info("Agent decided goal is satisfied. Terminating loop.")
                return response.content

            # Step 3: Actuation (Tool Execution)
            for tool_call in response.tool_calls:
                call_id = tool_call["id"]
                tool_name = tool_call["name"]
                args = tool_call["args"]

                if tool_name not in self.tool_map:
                    obs = f"Error: Tool '{tool_name}' not available."
                else:
                    try:
                        obs = str(self.tool_map[tool_name].invoke(args))
                    except Exception as e:
                        obs = f"Execution Error: {e}"

                # Append Observation back into state
                messages.append(ToolMessage(tool_call_id=call_id, content=obs))

        return "Agent aborted: Reached maximum iteration limit without resolving goal."
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Interactive ReAct Cognitive Loop Simulator", divider="rainbow")

st.markdown("""
Observe how an autonomous ReAct agent deconstructs an enterprise incident, decides which tools to invoke, 
evaluates intermediate observations, and converges on a grounded final answer.
""")

sim = MockReActAgentSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🎯 Configure Agent Goal")
    preset_goals = {
        "Investigate Checkout Latency Spike": "Customer ops-lead@enterprise.corp reported a severe latency spike during checkout.",
        "Verify Scheduled Maintenance Status": "Check if customer ops-lead@enterprise.corp was impacted by the scheduled maintenance window."
    }
    
    selected_preset = st.selectbox("Select Mission Objective", list(preset_goals.keys()))
    task_input = st.text_area("Agent Objective", value=preset_goals[selected_preset], height=100)
    max_turns_slider = st.slider("Max Permitted Reasoning Turns", min_value=2, max_value=8, value=5)

with col_wb2:
    st.subheader("⚙️ Agent Environment Specification")
    st.info("""
    **Cognitive Loop Parameters:**
    - **Architecture**: ReAct (Reason + Act + Observe)
    - **Registered Tools**:
      1. `query_customer_db(user_email)`
      2. `query_cluster_metrics(service, metric)`
      3. `fetch_support_ticket(ticket_id)`
    - **Stopping Condition**: Model emits plain text with `tool_calls=[]`
    """)

if st.button("🚀 Execute Autonomous Agent Loop", type="primary"):
    res = sim.run_agent_task(task_input, max_turns_slider)
    
    st.subheader("📊 ReAct Cognitive Trace")
    m1, m2 = st.columns(2)
    m1.metric("Reasoning Turns Taken", res["total_turns"])
    m2.metric("Execution State", res["status"])

    st.write("#### 🧠 Step-by-Step Thought, Action & Observation Trail:")
    for step in res["trace"]:
        with st.expander(f"Turn {step.step_number}: Action ➔ `{step.action}`", expanded=True):
            st.markdown(f"**💭 Thought**: *{step.thought}*")
            st.markdown(f"**⚡ Action**: `{step.action}` with input `{step.action_input}`")
            st.success(f"**👁️ Observation**: {step.observation}")

    st.write("#### 🎯 Grounded Final Resolution:")
    st.info(res["final_answer"])

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the fundamental architectural difference between an LCEL Chain and an AI Agent?",
     "An LCEL Chain is a directed acyclic graph (DAG) where the execution path is fixed by code at authoring time. An AI Agent is a cyclic state machine where the LLM dynamically chooses the execution path, tools, and termination condition at runtime based on incoming intermediate observations."),
    ("How does the ReAct framework improve reasoning reliability compared to direct action dispatch?",
     "ReAct forces the model to interleave verbal reasoning ('Thoughts') before taking programmatic actions ('Actions'). Generating a thought forces the model to self-reflect on past observations, plan the next sub-goal, and catch previous mistakes before emitting tool parameters, significantly reducing hallucination."),
    ("How does an agent determine its stopping condition?",
     "In modern tool calling architectures, the agent terminates when the model returns an AIMessage that contains pure text content and zero tool_calls. In production systems, this is supplemented by hard safeguards such as max_iterations, execution timeouts, and cycle detection."),
    ("Why is agent execution inherently non-deterministic compared to classical software?",
     "Because the LLM dynamically decides which tools to call and what arguments to pass based on non-deterministic token sampling and dynamic environment feedback. Two runs with the same objective might choose different sequences of tools to achieve the same result."),
    ("What are the primary latency and cost challenges associated with autonomous agents?",
     "Each step in an agent loop requires a complete LLM inference turn plus tool execution time. A 4-step ReAct agent will make 5 distinct LLM API calls, accumulating substantial token costs and latency (often 5 to 15 seconds). Agents must be designed with minimal turn budgets and parallel tool execution.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Runaway Infinite Loops. Deploying an agent without a strict max_iterations limit can cause the agent to call tools infinitely if an API repeatedly returns ambiguous results, exhausting API budgets within hours. Always enforce a hard turn limit (e.g. max_turns=6).")
    st.error("**Trap 2 (Breaking Error)**: Forgetting to Return Tool Observations to State. If a tool executes but the resulting ToolMessage is not appended to conversation history before the next model call, the agent will re-call the exact same tool endlessly.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Premature Tool Escalation. Allowing the agent to call heavy, expensive tools (like live web search or big data queries) before checking local cache memory wastes tokens and adds unnecessary latency.")
    st.warning("**Trap 4 (Quality Warning)**: Overly Vague User Goals. Passing open-ended prompts like 'Fix the server' without scoping constraints causes the agent to explore irrelevant system logs. Ground agent goals with clear boundaries.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Chain vs Agent Control"):
        st.markdown("**Chains have hard-coded execution paths; Agents dynamically decide paths at runtime.**")
    with st.expander("🎴 Flashcard 2: ReAct Core Cycle"):
        st.markdown("**Thought (Reasoning) ➔ Action (Tool Dispatch) ➔ Observation (Environment Result).**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Termination Signal"):
        st.markdown("**The model emits an AIMessage containing text content and an empty tool_calls list.**")
    with st.expander("🎴 Flashcard 4: max_iterations Safeguard"):
        st.markdown("**Hard circuit breaker preventing runaway token costs and infinite agent loops.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can articulate the architectural differences between an LCEL Chain and an Agent.", value=True)
chk2 = st.checkbox("I understand the Sense-Reason-Act perception loop governing agent execution.", value=True)
chk3 = st.checkbox("I understand the ReAct framework and the role of interleaved Thought steps.", value=True)
chk4 = st.checkbox("I know how models select tools based on descriptions and schema alignment.", value=True)
chk5 = st.checkbox("I can configure stopping conditions and max_iterations safeguards.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 agent fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 9.1 Agent Fundamentals. Ready for 9.2 Creating Agents!")
