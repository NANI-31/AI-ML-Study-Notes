import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="20.1 Multi-Agent Architecture - LangGraph Curriculum",
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
    .badge-agent {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-coord {
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
class MultiAgentArchitectureSimulator:
    """
    Simulates single-agent vs multi-agent execution lifecycles:
    - Context window load & prompt drift
    - Specialist agent task allocation
    - Coordination latency vs output quality metrics
    """
    def __init__(self):
        pass

    def compare_architectures(self, task_complexity: str, num_subtasks: int) -> Dict[str, Any]:
        if task_complexity == "Low (Single Domain)":
            single_tokens = 1200
            single_accuracy = 0.94
            single_latency_sec = 1.4
            
            multi_tokens = 2800
            multi_accuracy = 0.95
            multi_latency_sec = 2.8
        elif task_complexity == "Medium (2-3 Domains)":
            single_tokens = 4500
            single_accuracy = 0.78
            single_latency_sec = 4.2
            
            multi_tokens = 6200
            multi_accuracy = 0.92
            multi_latency_sec = 4.8
        else: # High (Cross-functional Enterprise)
            single_tokens = 14000
            single_accuracy = 0.54  # Severe prompt drift / context overload
            single_latency_sec = 12.0
            
            multi_tokens = 11500  # Partitioned specialist contexts
            multi_accuracy = 0.91
            multi_latency_sec = 6.5  # Parallel worker superstep saves time

        return {
            "single": {
                "tokens": single_tokens,
                "accuracy": single_accuracy,
                "latency_sec": single_latency_sec,
                "failure_mode": "None" if single_accuracy > 0.8 else "Prompt Drift & Context Degradation"
            },
            "multi": {
                "tokens": multi_tokens,
                "accuracy": multi_accuracy,
                "latency_sec": multi_latency_sec,
                "coordination_overhead_sec": 0.6,
                "specialists_used": num_subtasks
            }
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Multi-Agent Systems, Topologies, and Coordination Dynamics", divider="rainbow")

st.markdown("""
As real-world enterprise applications expand in complexity, attempting to solve an entire problem with a single omniscient agent leads to failure.
A single agent forced to juggle web search, SQL execution, Python analysis, financial modeling, and customer tone guidelines inevitably suffers from:
- **Prompt Overload & Tool Hallucination**: When 20+ tool definitions are bound to a single LLM, tool selection accuracy drops precipitously.
- **Context Window Saturation**: Intermediate tool outputs fill the context window, causing needle-in-a-haystack retrieval loss.
- **Fragile Recovery**: A bug in an analytical loop crashes the entire session.

A **Multi-Agent System (MAS)** in LangGraph decomposes complex objectives into networks of specialized, autonomous agents.
Each specialist maintains a narrow persona, an isolated toolset, and a dedicated execution loop.
An orchestration layer coordinates task distribution, message routing, and deliverable synthesis.

The blueprint below illustrates the core topology of a modern multi-agent architecture in LangGraph.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiAgentTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_ingress {
            label="1. Ingress & Supervisor Orchestrator";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_client [label="User / API Request\\n(High-Level Complex Objective)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            supervisor_agent [label="Supervisor Agent\\n(Decomposes Task & Evaluates Handoffs)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_specialists {
            label="2. Autonomous Specialist Agents (Dedicated Loops & Tools)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            researcher [label="Research Specialist\\nTools: [Search, ArXiv]\\nContext: Minimal Focused", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            analyst [label="Data Analyst Specialist\\nTools: [Python REPL, SQL]\\nContext: Numeric Data Only", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            writer [label="Report Writer Specialist\\nTools: [Docx, MarkdownFormat]\\nContext: Editorial Synthesis", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_communication_bus {
            label="3. Communication & State Coordination Bus";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            state_bus [label="Shared State Message Bus\\n(TypedDict with Channel Reducers)", fillcolor="#18181B", fontcolor="#E4E4E7", color="#71717A"];
            validator [label="Validation / Quality Gate Node\\nAsserts Deliverable Criteria", fillcolor="#701A75", fontcolor="#F5D0FE", color="#E879F9"];
        }

        user_client -> supervisor_agent [label="Initiates Thread"];
        supervisor_agent -> researcher [label="Delegates Gathering", color="#60A5FA"];
        supervisor_agent -> analyst [label="Delegates Metrics", color="#60A5FA"];
        
        researcher -> state_bus [label="Yields Findings", color="#818CF8"];
        analyst -> state_bus [label="Yields Calculations", color="#F97316"];
        
        state_bus -> supervisor_agent [label="Aggregates Progress"];
        supervisor_agent -> writer [label="Dispatches Draft Assembly", color="#34D399"];
        writer -> validator [label="Draft Submitted"];
        validator -> supervisor_agent [label="Needs Revision", color="#EF4444", style=dashed];
        validator -> user_client [label="Approved Final Response", color="#34D399"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: WHY MULTI-AGENT SYSTEMS?
# ==============================================================================
st.header("2. Why Multi-Agent Systems?: Overcoming Context Saturation, Prompt Drift, and Single-Agent Bottlenecks", divider="rainbow")

st.markdown("""
Understanding why production systems migrate to multi-agent architectures requires analyzing the three fatal flaws of single-agent designs:
""")

col_wh1, col_wh2, col_wh3 = st.columns(3)
with col_wh1:
    st.markdown("#### 1. Context Saturation")
    st.markdown("""
    When an agent performs 5 sequential web searches and runs 3 Python scripts, its token window fills with thousands of lines of raw text.
    As tokens accumulate, LLM reasoning degrades exponentially (the **lost-in-the-middle** effect).
    """)

with col_wh2:
    st.markdown("#### 2. Prompt Drift")
    st.markdown("""
    A system prompt packed with 40 instructions ("be polite, use SQL, check security, format in APA style, execute unit tests") causes instruction competition.
    The LLM inevitably forgets constraints or compromises quality on sub-tasks.
    """)

with col_wh3:
    st.markdown("#### 3. Tool Choice Confusion")
    st.markdown("""
    Binding 25 tools to one model degrades function calling accuracy.
    The model frequently picks the wrong tool, hallucinates parameter names, or loops between unrelated capabilities.
    """)

# ==============================================================================
# SECTION 3: SINGLE-AGENT VS MULTI-AGENT
# ==============================================================================
st.header("3. Single-Agent vs Multi-Agent: Comparative Trade-Offs, Failure Modes, and Scaling Laws", divider="rainbow")

st.markdown("""
Multi-agent systems provide superior modularity and reasoning depth, but introduce coordination latency and communication complexity.
Architects must evaluate the engineering trade-offs before choosing between single-agent and multi-agent designs:
""")

df_compare = pd.DataFrame({
    "Dimension": [
        "Prompt Complexity",
        "Tool Count per Model",
        "Context Isolation",
        "Failure Blast Radius",
        "Execution Concurrency",
        "Debugging Observability",
        "Token Cost for Simple Tasks"
    ],
    "Single-Agent Design": [
        "Monolithic (50+ instructions)",
        "Unbounded (15 to 30+ tools)",
        "Zero (all scratchpad data shared)",
        "High (single error aborts thread)",
        "Strictly Sequential",
        "Opaque (interleaved reasoning)",
        "Low (single turn)"
    ],
    "Multi-Agent System (LangGraph)": [
        "Modular (focused persona per agent)",
        "Strictly bounded (1 to 4 tools per agent)",
        "High (private states or filtered busses)",
        "Isolated (specialist retry loops)",
        "High (parallel fan-out supersteps)",
        "Transparent (traceable per-agent spans)",
        "Moderate (coordination message overhead)"
    ]
})
st.dataframe(df_compare, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 4: AGENT SPECIALIZATION
# ==============================================================================
st.header("4. Agent Specialization: Persona Framing, Constrained Toolsets, and High-Fidelity Domain Prompts", divider="rainbow")

st.markdown("""
The foundation of an effective multi-agent system is **high-fidelity specialization**.
Instead of generalist instructions, each agent node is configured with:
1. **Strict Persona Framing**: Explicit declaration of the agent's singular identity, perspective, and deliverable format.
2. **Constrained Tool Binding**: Providing only the 1-3 tools directly required for that agent's core function.
3. **Dedicated Output Schemas**: Structured output enforcement via Pydantic or TypedDict schemas.
""")

col_sp1, col_sp2 = st.columns(2)
with col_sp1:
    st.markdown("#### Anti-Pattern: The Overloaded Generalist")
    st.code('''# ANTI-PATTERN: Generalist model trying to do everything
generalist_model = ChatOpenAI(model="gpt-4o").bind_tools([
    search_web, execute_sql, run_python,
    send_slack, create_jira, format_pdf,
    query_vector_db, calculate_tax
])''', language="python")

with col_sp2:
    st.markdown("#### Production Pattern: Clean Specialized Agents")
    st.code('''# PRODUCTION PATTERN: Dedicated specialist agents
sql_specialist = ChatOpenAI(model="gpt-4o").bind_tools([execute_sql])
python_analyst = ChatOpenAI(model="gpt-4o").bind_tools([run_python])
researcher_agent = ChatOpenAI(model="gpt-4o-mini").bind_tools([search_web])

# Each agent runs inside its own focused node function!''', language="python")

# ==============================================================================
# SECTION 5: AGENT COMMUNICATION
# ==============================================================================
st.header("5. Agent Communication: Message Passing, Handoff Protocols, and Shared Context Busses", divider="rainbow")

st.markdown("""
Agents in LangGraph communicate primarily via **state channel updates**.
There are three standard communication protocols in multi-agent engineering:
""")

st.markdown("""
1. **Shared History Bus (`MessagesState`)**:
   - All agents read from and append to a centralized `messages` channel using `add_messages`.
   - Each agent prefixes its messages with its persona name (`AIMessage(content="...", name="researcher")`).
   - Downstream agents inspect message provenance to understand which specialist produced each finding.

2. **Structured Artifact Handoffs**:
   - Agents do not share raw conversational chatter.
   - Instead, Agent A populates a dedicated channel (e.g., `research_dossier`), and Agent B consumes only that structured channel.

3. **Explicit Handoff Tool Calls (`transfer_to_agent`)**:
   - The LLM emits a tool call representing a transfer of authority: `transfer_to_sql_specialist(query="...")`.
   - The conditional router edge directs execution to the targeted specialist node immediately.
""")

# ==============================================================================
# SECTION 6: AGENT COORDINATION
# ==============================================================================
st.header("6. Agent Coordination: Centralized Orchestration vs Decentralized Choreography", divider="rainbow")

st.markdown("""
Multi-agent systems coordinate execution through two fundamental paradigms:
""")

col_co1, col_co2 = st.columns(2)
with col_co1:
    st.markdown("#### Centralized Orchestration (Supervisor Pattern)")
    st.markdown("""
    - A designated **Supervisor Node** receives user inputs and delegates work to specialist nodes.
    - After every specialist finishes, execution routes back to the supervisor.
    - **Pros**: Clear control flow, deterministic stopping conditions, easy debugging.
    - **Cons**: Supervisor LLM must execute after every step, adding token cost and latency.
    """)

with col_co2:
    st.markdown("#### Decentralized Choreography (Peer Handoffs)")
    st.markdown("""
    - Agents transition directly to peer agents via conditional edges without returning to a central supervisor.
    - Agent A finishes research and hands off directly to Agent B (Analyst).
    - **Pros**: Lower latency, reduced supervisor LLM token consumption.
    - **Cons**: Prone to infinite ping-pong loops if stopping conditions are poorly defined.
    """)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Multi-Agent Coordination Graph", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a multi-agent system with a Supervisor Orchestrator, two specialized worker nodes, and clean message routing.
""")

st.code('''from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition with Provenance Tracking
class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: Literal["researcher", "coder", "FINISH"]
    final_output: str

# 2. Specialist Worker Nodes
def researcher_node(state: MultiAgentState) -> dict:
    last_msg = state["messages"][-1].content
    # Specialist performs research and labels message with name
    finding = f"Research findings on: '{last_msg}'. Data points verified: 3 sources."
    return {
        "messages": [AIMessage(content=finding, name="researcher")]
    }

def coder_node(state: MultiAgentState) -> dict:
    last_msg = state["messages"][-1].content
    # Specialist writes Python code
    code = f"def solution():\\n    # Generated based on: {last_msg}\\n    return 42"
    return {
        "messages": [AIMessage(content=code, name="coder")]
    }

# 3. Supervisor Orchestration Node
def supervisor_node(state: MultiAgentState) -> dict:
    messages = state["messages"]
    # In production, an LLM evaluates messages and outputs a structured choice
    has_research = any(getattr(m, "name", "") == "researcher" for m in messages)
    has_code = any(getattr(m, "name", "") == "coder" for m in messages)
    
    if not has_research:
        return {"next_agent": "researcher"}
    elif not has_code:
        return {"next_agent": "coder"}
    else:
        return {"next_agent": "FINISH", "final_output": "Task successfully solved by specialist team."}

# 4. Routing Function
def route_next_step(state: MultiAgentState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

# 5. Build Graph
builder = StateGraph(MultiAgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_next_step,
    {
        "researcher": "researcher",
        "coder": "coder",
        END: END
    }
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

app = builder.compile(checkpointer=MemorySaver())

# Execute
res = app.invoke(
    {"messages": [HumanMessage(content="Build an automated volatility trading monitor")]},
    config={"configurable": {"thread_id": "mas_demo_01"}}
)
print("Execution Complete! Messages count:", len(res["messages"]))
print("Final Summary:", res["final_output"])''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Multi-Agent Coordination Simulator & Context Load Analyzer", divider="rainbow")

st.markdown("""
Compare single-agent vs multi-agent execution dynamics across different complexity levels and analyze token load and error risks.
""")

sim = MultiAgentArchitectureSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ Simulation Controls")
    task_complexity = st.selectbox(
        "Task Complexity Level",
        ["Low (Single Domain)", "Medium (2-3 Domains)", "High (Cross-functional Enterprise)"]
    )
    num_subtasks = st.slider("Number of Sub-Tasks / Specialists", min_value=1, max_value=6, value=3)

    if st.button("🚀 Run Architecture Comparison"):
        results = sim.compare_architectures(task_complexity, num_subtasks)
        st.session_state["mas_sim_results"] = results

with col_sim2:
    st.subheader("📊 Comparative Performance Audit")
    if "mas_sim_results" in st.session_state:
        res = st.session_state["mas_sim_results"]
        s = res["single"]
        m = res["multi"]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Single-Agent Accuracy", f"{int(s['accuracy'] * 100)}%", delta=f"{s['latency_sec']}s Latency")
            st.metric("Single-Agent Context Tokens", f"{s['tokens']:,}")
            if s["failure_mode"] != "None":
                st.error(f"⚠️ Risk: {s['failure_mode']}")
            else:
                st.success(" Operational")

        with col_m2:
            st.metric("Multi-Agent Accuracy", f"{int(m['accuracy'] * 100)}%", delta=f"{m['latency_sec']}s Latency")
            st.metric("Multi-Agent Context Tokens", f"{m['tokens']:,}")
            st.success(f" Specialized Across {m['specialists_used']} Agents")

        df_bench = pd.DataFrame([
            {"Architecture": "Single Monolithic Agent", "Accuracy": f"{s['accuracy']*100:.1f}%", "Tokens": s["tokens"], "Latency (s)": s["latency_sec"]},
            {"Architecture": "LangGraph Multi-Agent", "Accuracy": f"{m['accuracy']*100:.1f}%", "Tokens": m["tokens"], "Latency (s)": m["latency_sec"]}
        ])
        st.dataframe(df_bench, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the primary motivation for migrating from a single agent to a multi-agent system?", expanded=False):
    st.markdown("""
    **Answer**:
    Single agents suffer from context window saturation, prompt drift, and tool selection degradation when tasked with cross-functional objectives requiring 10+ tools.
    Multi-agent systems partition complexity into specialized nodes with isolated contexts, dedicated personas, and focused toolsets.
    """)

with st.expander("Q2: How does LangGraph prevent duplicate messages when multiple agents converse?", expanded=False):
    st.markdown("""
    **Answer**:
    By using the `add_messages` reducer on the state messages channel.
    Each agent returns only newly generated `AIMessage` objects with their specific `name` attribute attached.
    The reducer merges them cleanly into the history array without overwriting existing messages.
    """)

with st.expander("Q3: What are the trade-offs between a centralized Supervisor and decentralized Peer Handoffs?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Supervisor**: Deterministic control, clean stopping conditions, easier observability, but higher token cost and latency because the supervisor executes between every step.
    - **Peer Handoffs**: Lower latency and lower token cost, but prone to circular routing loops and harder to debug.
    """)

with st.expander("Q4: How do you identify which agent generated an AI message in a multi-agent chat channel?", expanded=False):
    st.markdown("""
    **Answer**:
    By setting the `name` attribute on the message object:
    `AIMessage(content="...", name="financial_analyst")`.
    Downstream nodes and frontends can inspect `message.name` to route responses and render customized UI avatars.
    """)

with st.expander("Q5: Can multi-agent systems execute specialist tasks in parallel in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! By configuring multiple edges originating from the supervisor or using the `Send` API to map tasks to multiple specialist nodes simultaneously, LangGraph executes all specialist nodes concurrently in the same superstep.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Ping-Pong Loops Between Peer Agents")
    st.markdown("""
    *Mechanism*: Agent A transfers to Agent B, which transfers back to Agent A without an exit condition.  
    *Consequence*: Graph execution spirals until hitting the recursion limit, burning tokens and timing out.  
    *Fix*: Always enforce a `max_turns` counter in state and set `recursion_limit` on graph invocation.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing message.name Leading to Identity Confusion")
    st.markdown("""
    *Mechanism*: Specialist agents returning bare `AIMessage(content="...")` without assigning the `name` attribute.  
    *Consequence*: The LLM supervisor cannot determine which agent produced which finding and repeats tasks.  
    *Fix*: Always pass `name="agent_identifier"` when instantiating `AIMessage`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Context Leakage via Monolithic Shared State")
    st.markdown("""
    *Mechanism*: Putting 500KB of raw HTML from a research agent into the shared `messages` channel.  
    *Consequence*: Forces every subsequent agent (coder, writer) to pay token costs for irrelevant HTML.  
    *Fix*: Store raw research in a private child state or dedicated scratchpad channel, passing only summaries.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Over-Supervising Simple Sequential Pipelines")
    st.markdown("""
    *Mechanism*: Calling an LLM supervisor between two tasks that are always sequential (e.g., fetch data -> format data).  
    *Consequence*: Adds 1.5 seconds of unnecessary LLM latency and cost to every request.  
    *Fix*: Use deterministic code routing or direct edges when the next step is fixed.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Multi-Agent System**\\nA network of specialized autonomous agents coordinated by an orchestration graph.")
with col_fc2:
    st.info("**Supervisor Pattern**\\nA centralized orchestrator node that evaluates state and delegates to specialist workers.")
with col_fc3:
    st.info("**message.name**\\nThe metadata attribute identifying which specialist produced a specific AIMessage.")
with col_fc4:
    st.info("**Prompt Drift**\\nDegradation in instruction adherence when an LLM is overloaded with competing tasks.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural reasons for using multi-agent systems over monolithic agents.", value=True)
c2 = st.checkbox("I know the trade-offs between single-agent and multi-agent systems regarding cost and latency.", value=True)
c3 = st.checkbox("I understand how agent specialization prevents context saturation and tool selection errors.", value=True)
c4 = st.checkbox("I know how to use the message.name attribute for message provenance tracking.", value=True)
c5 = st.checkbox("I can contrast centralized supervisor orchestration with decentralized peer handoffs.", value=True)
c6 = st.checkbox("I know how to prevent infinite ping-pong loops using recursion limits and max_turns counters.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 20.1 'Multi-Agent Architecture' with full architectural and practical comprehension!")
